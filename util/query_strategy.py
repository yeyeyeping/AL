import albumentations as A
from os.path import join
import random
from pymic.util.evaluation_seg import binary_dice
import numpy as np
import torch
from torch.utils.data import DataLoader
import util.jitfunc as f
from torch import nn
from sklearn.metrics import pairwise_distances
from torch.nn import functional as F
from util import SubsetSampler
from util.taalhelper import augments_forward
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.covariance import EmpiricalCovariance
from sklearn.mixture import GaussianMixture
import random
from util import mode_filter
import copy

# todo: 前向过程应由trainer负责，这样可以避免不同模型输出不一样的问题
# 两步选取记录中间过程


class LimitSortedList(object):

    def __init__(self, limit, descending=False) -> None:
        self.descending = descending
        self.limit = limit
        self._data = []

    def reset(self):
        self._data.clear()

    @property
    def data(self):
        return map(lambda x: int(x[0]), self._data)

    def extend(self, idx_score):
        assert isinstance(idx_score, (torch.Tensor, np.ndarray, list, tuple))
        idx_score = list(idx_score)
        self._data.extend(idx_score)
        if len(self._data) > self.limit:
            self._data = sorted(self._data,
                                key=lambda x: x[1],
                                reverse=self.descending)[:self.limit]


class QueryStrategy(object):

    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__()
        train_dataset = copy.deepcopy(dataloader["unlabeled"].dataset)
        train_dataset.transforms = None
        # self.unlabeled_dataloader = dataloader["unlabeled"]
        # self.labeled_dataloader = dataloader["labeled"]
        self.unlabeled_dataloader = DataLoader(
            train_dataset,
            batch_size=16,
            sampler=dataloader["unlabeled"].sampler,
            persistent_workers=True,
            prefetch_factor=4,
            generator=dataloader["unlabeled"].generator,
            num_workers=4,
        )

        self.labeled_dataloader = DataLoader(
            train_dataset,
            batch_size=16,
            sampler=dataloader["labeled"].sampler,
            persistent_workers=True,
            prefetch_factor=4,
            generator=dataloader["labeled"].generator,
            num_workers=4,
        )

    def select_dataset_idx(self, query_num):
        raise NotImplementedError

    def convert2img_idx(self, ds_idx, dataloader: DataLoader):
        return [dataloader.sampler.indices[img_id] for img_id in ds_idx]

    def sample(self, query_num):
        ret = self.select_dataset_idx(query_num)
        if isinstance(ret,tuple)  and len(ret)== 2:
            img_idx = self.convert2img_idx(*ret)
        else:
            img_idx = self.convert2img_idx(ret, self.unlabeled_dataloader)
        self.labeled_dataloader.sampler.indices.extend(img_idx)
        # 注意这里不可以用index来移除，因为pop一个之后，原数组就变换了
        # for i in dataset_idx:
        #     self.unlabeled_dataloader.sampler.indices.pop(i)
        for item in img_idx:
            self.unlabeled_dataloader.sampler.indices.remove(item)


class RandomQuery(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)

    def sample(self, query_num):
        np.random.shuffle(self.unlabeled_dataloader.sampler.indices)
        self.labeled_dataloader.sampler.indices.extend(
            self.unlabeled_dataloader.sampler.indices[:query_num])
        del self.unlabeled_dataloader.sampler.indices[:query_num]


class SimpleQueryStrategy(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        assert "descending" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.descending = kwargs["descending"]

    def compute_score(self, model_output):
        raise NotImplementedError

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        q = LimitSortedList(limit=query_num, descending=self.descending)
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(device)
            output = self.model(img)
            score = self.compute_score(output).cpu()
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack(
                [torch.arange(offset, offset + len(img)), score])
            q.extend(idx_entropy.tolist())
        return q.data


class MaxEntropy(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=True, **kwargs)

    def compute_score(self, model_output):
        model_output, _, _ = model_output
        if len(model_output) == 1:
            o = model_output[0]
        elif len(model_output) > 1:
            o = model_output.mean(0)
        else:
            raise NotImplementedError
        return f.max_entropy(o.softmax(dim=1))


class MarginConfidence(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        model_output, _, _ = model_output
        if len(model_output) == 1:
            o = model_output[0]
        elif len(model_output) > 1:
            o = model_output.mean(0)
        else:
            raise NotImplementedError
        return f.margin_confidence(o.softmax(dim=1))


class CircuAreaSample(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=True, **kwargs)

    def compute_score(self, model_output):
        model_output, _, _ = model_output
        if len(model_output) == 1:
            o = model_output[0]
        elif len(model_output) > 1:
            o = model_output.mean(0)
        else:
            raise NotImplementedError
        o = o.argmax(dim=1)

        return f.car(o)


class LeastConfidence(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        model_output, _, _ = model_output
        if len(model_output) == 1:
            o = model_output[0]
        elif len(model_output) > 1:
            o = model_output.mean(0)
        else:
            raise NotImplementedError
        return f.least_confidence(o.softmax(dim=1))


class SNDQuery(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=False, **kwargs)

    def compute_score(self, model_output):
        model_output, _, _ = model_output
        if len(model_output) == 1:
            o = model_output[0]
        elif len(model_output) > 1:
            o = model_output.mean(0)
        else:
            raise NotImplementedError
        return f.snd(o.softmax(1))


class OnlineMGQuery(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=True, **kwargs)

    @torch.no_grad()
    def compute_score(self, model_output):
        output = torch.stack(model_output[0])
        return f.JSD(output.softmax(2))


# ssssssssssssss


class TAALQuery(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.model = kwargs["trainer"].model
        self.num_augmentations = int(kwargs.get("num_augmentations", 10))

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        q = LimitSortedList(limit=query_num, descending=True)
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(device)
            output, _, _ = self.model(img)
            aug_out = augments_forward(
                img,
                self.model,
                output[0].softmax(dim=1),
                self.num_augmentations,
                device,
            )
            score = f.JSD(aug_out).cpu()
            assert score.shape[0] == img.shape[0], "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack(
                [torch.arange(offset, offset + img.shape[0]), score])
            q.extend(idx_entropy)

        return q.data


class MaskedFeatureCoresetQuery(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.adpative_pool = nn.AdaptiveAvgPool2d(output_size=(128, 128))
        self.max_pool = nn.MaxPool2d(128, 128)

    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            output, _, features = self.model(img)
            last_feature = features[-1]
            prob = torch.stack(output).mean(0)
            masked = self.adpative_pool(prob).argmax(1)
            embedding = self.max_pool(last_feature * masked[:, None])
            embedding_list.append(embedding.view(embedding.shape[:2]))
        return torch.concat(embedding_list, dim=0)

    def select_dataset_idx(self, query_num):
        self.model.eval()
        embedding_unlabeled = self.embedding(self.unlabeled_dataloader)
        embedding_labeled = self.embedding(self.labeled_dataloader)
        return self.furthest_first(
            unlabeled_set=embedding_unlabeled.cpu().numpy(),
            labeled_set=embedding_labeled.cpu().numpy(),
            budget=query_num,
        )

    def furthest_first(self, unlabeled_set, labeled_set, budget):
        """
        Selects points with maximum distance

        Parameters
        ----------
        unlabeled_set: numpy array
            Embeddings of unlabeled set
        labeled_set: numpy array
            Embeddings of labeled set
        budget: int
            Number of points to return
        Returns
        ----------
        idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """
        m = np.shape(unlabeled_set)[0]
        if np.shape(labeled_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(unlabeled_set, labeled_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for _ in range(budget):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(unlabeled_set,
                                              unlabeled_set[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs


class CoresetQuery(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model

        pool_size = int(kwargs.get("pool_size", 12))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        self.multi_level_feature = bool(
            kwargs.get("multi_level_feature", False))
        self.pool.eval()

    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            _, _, features = self.model(img)
            if self.multi_level_feature:
                embedding = torch.concat([self.pool(i) for i in features],
                                         dim=1).view((img.shape[0], -1))
            else:
                embedding = self.pool(features[0]).view((img.shape[0], -1))
            embedding_list.append(embedding)
        return torch.concat(embedding_list, dim=0)

    def select_dataset_idx(self, query_num):
        self.model.eval()
        embedding_unlabeled = self.embedding(self.unlabeled_dataloader)
        embedding_labeled = self.embedding(self.labeled_dataloader)
        return self.furthest_first(
            unlabeled_set=embedding_unlabeled.cpu().numpy(),
            labeled_set=embedding_labeled.cpu().numpy(),
            budget=query_num,
        )

    def furthest_first(self, unlabeled_set, labeled_set, budget):
        """
        Selects points with maximum distance

        Parameters
        ----------
        unlabeled_set: numpy array
            Embeddings of unlabeled set
        labeled_set: numpy array
            Embeddings of labeled set
        budget: int
            Number of points to return
        Returns
        ----------
        idxs: list
            List of selected data point indexes with respect to unlabeled_x
        """
        m = np.shape(unlabeled_set)[0]
        if np.shape(labeled_set)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(unlabeled_set, labeled_set)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for _ in range(budget):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(unlabeled_set,
                                              unlabeled_set[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs


class CoresetConsistencyQuery(CoresetQuery):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = int(kwargs.get("alpha", 2))
        self.dataset = self.unlabeled_dataloader.dataset
        self.device = next(iter(self.model.parameters())).device

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        qn = query_num * self.alpha
        dataset_idx = super().select_dataset_idx(qn)
        img_idx = self.convert2img_idx(dataset_idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        q = LimitSortedList(limit=query_num, descending=True)
        self.model.eval()
        for batch_idx, (img, _) in enumerate(aux_dataloader):
            img = img.to(self.device)
            pred, _, _ = self.model(img)
            score = f.JSD(torch.stack(pred).softmax(2))
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * aux_dataloader.batch_size
            idx_score = torch.column_stack(
                [torch.arange(offset, offset + len(img)),
                 score.cpu()])
            q.extend(idx_score)
        return q.data


class AverageAlignConsistency(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=True, **kwargs)
        self.device = next(iter(self.model.parameters())).device

    @torch.no_grad()
    def compute_score(self, model_output):
        model_output = torch.stack(model_output[0]).softmax(2)
        avg_pred = torch.mean(model_output, dim=0) * 0.99 + 0.005
        consistency = torch.zeros(len(model_output[1]), device=self.device)
        for aux in model_output:
            aux = aux * 0.99 + 0.005
            var = torch.sum(
                nn.functional.kl_div(aux.log(), avg_pred, reduction="none"),
                dim=1,
                keepdim=True,
            )
            exp_var = torch.exp(-var)
            square_e = torch.square(avg_pred - aux)
            c = torch.mean(square_e * exp_var, dim=[-1, -2, -3]) / (
                torch.mean(exp_var, dim=[-1, -2, -3]) + 1e-8)
            consistency += c
        return consistency


class StochasticBatch(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.model = kwargs["trainer"].model
        self.device = next(iter(self.model.parameters())).device

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        uncertainty_score = []
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(self.device)
            output = self.model(img)[0][0]
            score = f.max_entropy(output.softmax(dim=1)).cpu()
            assert score.shape[0] == img.shape[0], "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack(
                [torch.arange(offset, offset + img.shape[0]), score])
            uncertainty_score.extend(idx_entropy.tolist())
        random.shuffle(uncertainty_score)
        # todo:Is that  necessary to drop last few samplers?
        uncertainty_score = np.asanyarray(uncertainty_score)
        splits = np.split(
            uncertainty_score,
            [s for s in range(query_num, len(uncertainty_score), query_num)],
        )
        batch_uncertainty = list(map(lambda x: x[:, 1].sum(), splits))
        max_idx = np.argmax(batch_uncertainty)
        selected_batch = splits[max_idx]
        return selected_batch[:, 0].astype(np.uint64)


class BALD(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.model = kwargs["trainer"].model
        self.dropout_round = int(kwargs.get("round", 10))
        self.device = next(iter(self.model.parameters())).device

    def dropout_on(self):
        self.model.drop1 = nn.Dropout(p=0.1)
        self.model.drop2 = nn.Dropout(p=0.2)
        self.model.drop3 = nn.Dropout(p=0.3)
        self.model.drop4 = nn.Dropout(p=0.4)
        self.model.drop5 = nn.Dropout(p=0.5)

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        self.dropout_on()
        q = LimitSortedList(limit=query_num, descending=True)
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            out = torch.empty(
                self.dropout_round,
                img.shape[0],
                2,
                img.shape[-2],
                img.shape[-1],
                device=self.device,
            )
            for round_ in range(self.dropout_round):
                img = img.to(self.device)
                output = self.model(img)[0][0]
                out[round_] = output.softmax(dim=1)
            score = f.JSD(out).cpu()
            assert score.shape[0] == img.shape[0], "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack(
                [torch.arange(offset, offset + img.shape[0]), score])
            q.extend(idx_entropy)

        return q.data


class EntropyConsistencyQuery(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=True, **kwargs)

    @torch.no_grad()
    def compute_score(self, model_output):
        output = torch.stack(model_output[0]).softmax(2)
        entropy = -output * torch.log(output + 1e-7)
        return f.JSD(entropy)


# 父类clustersample


class KmeansSample(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model

        pool_size = int(kwargs.get("pool_size", 12))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        self.multi_level_feature = bool(
            kwargs.get("multi_level_feature", False))
        self.pool.eval()

    # todo：应该是编码一个batch的feature，方便复用
    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            _, _, features = self.model(img)
            if self.multi_level_feature:
                embedding = torch.concat([self.pool(i) for i in features],
                                         dim=1).view((img.shape[0], -1))
            else:
                embedding = self.pool(features[0]).view((img.shape[0], -1))
            embedding_list.append(embedding)
        return torch.concat(embedding_list, dim=0)

    def select_dataset_idx(self, query_num):
        self.model.eval()
        features = self.embedding(self.unlabeled_dataloader)
        mkmeans = KMeans(query_num, n_init=10)
        distance = mkmeans.fit_transform(features.cpu().numpy())
        return np.unique(np.argmin(distance, axis=0))


class KmeansplusplusSample(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model

        pool_size = int(kwargs.get("pool_size", 12))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        self.multi_level_feature = bool(
            kwargs.get("multi_level_feature", False))
        self.pool.eval()

    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            _, _, features = self.model(img)
            if self.multi_level_feature:
                embedding = torch.concat([self.pool(i) for i in features],
                                         dim=1).view((img.shape[0], -1))
            else:
                embedding = self.pool(features[0]).view((img.shape[0], -1))
            embedding_list.append(embedding)
        return torch.concat(embedding_list, dim=0)

    def select_dataset_idx(self, query_num):
        self.model.eval()
        embedding_unlabeled = self.embedding(self.unlabeled_dataloader)
        mkmeans = KMeans(query_num, init="k-means++", n_init="auto")
        distance = mkmeans.fit_transform(embedding_unlabeled.cpu().numpy())
        return np.unique(np.argmin(distance, axis=0))


class MaskedKmeansSample(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.adpative_pool = nn.AdaptiveAvgPool2d(output_size=(128, 128))
        self.max_pool = nn.MaxPool2d(128, 128)

    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            output, _, features = self.model(img)
            last_feature = features[-1]
            prob = torch.stack(output).mean(0)
            masked = self.adpative_pool(prob).argmax(1)
            embedding = self.max_pool(last_feature * masked[:, None])
            embedding_list.append(embedding.view(embedding.shape[:2]))
        return torch.concat(embedding_list, dim=0)

    def select_dataset_idx(self, query_num):
        self.model.eval()
        embedding_unlabeled = self.embedding(self.unlabeled_dataloader)
        mkmeans = KMeans(query_num, n_init=10)
        distance = mkmeans.fit_transform(embedding_unlabeled.cpu().numpy())
        return np.unique(np.argmin(distance, axis=0))


class KmeansConsistencyQuery(KmeansSample):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = int(kwargs.get("alpha", 2))
        self.dataset = self.unlabeled_dataloader.dataset
        self.device = next(iter(self.model.parameters())).device

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        qn = query_num * self.alpha
        dataset_idx = super().select_dataset_idx(qn)
        img_idx = self.convert2img_idx(dataset_idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        q = LimitSortedList(limit=query_num, descending=True)
        self.model.eval()
        for batch_idx, (img, _) in enumerate(aux_dataloader):
            img = img.to(self.device)
            pred, _, _ = self.model(img)
            score = f.JSD(torch.stack(pred).softmax(2))
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * aux_dataloader.batch_size
            idx_score = torch.column_stack(
                [torch.arange(offset, offset + len(img)),
                 score.cpu()])
            q.extend(idx_score)
        return q.data, aux_dataloader


class KmeansVarQuery(KmeansSample):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = int(kwargs.get("alpha", 2))
        self.dataset = self.unlabeled_dataloader.dataset
        self.device = next(iter(self.model.parameters())).device

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        qn = query_num * self.alpha
        dataset_idx = super().select_dataset_idx(qn)
        img_idx = self.convert2img_idx(dataset_idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        q = LimitSortedList(limit=query_num, descending=True)
        self.model.eval()
        for batch_idx, (img, _) in enumerate(aux_dataloader):
            img = img.to(self.device)
            pred, _, _ = self.model(img)
            score = f.var(torch.stack(pred))
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * aux_dataloader.batch_size
            idx_score = torch.column_stack(
                [torch.arange(offset, offset + len(img)),
                 score.cpu()])
            q.extend(idx_score)
        return q.data, aux_dataloader


# todo:多继承


class VarKmeansQuery(AverageAlignConsistency):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = int(kwargs.get("alpha", 2))
        self.dataset = self.unlabeled_dataloader.dataset

        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model

        pool_size = int(kwargs.get("pool_size", 12))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        self.multi_level_feature = bool(
            kwargs.get("multi_level_feature", False))
        self.pool.eval()

    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for idx, (img, _) in enumerate(dataloader):
            img = img.to(device)
            _, _, features = self.model(img)
            if self.multi_level_feature:
                embedding = torch.concat([self.pool(i) for i in features],
                                         dim=1).view((img.shape[0], -1))
            else:
                embedding = self.pool(features[0]).view((img.shape[0], -1))
            embedding_list.append(embedding)
        return torch.concat(embedding_list, dim=0)

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        qn = query_num * self.alpha
        dataset_idx = super().select_dataset_idx(qn)
        img_idx = self.convert2img_idx(dataset_idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        self.model.eval()
        features = self.embedding(aux_dataloader)
        mkmeans = KMeans(query_num, n_init=10)
        distance = mkmeans.fit_transform(features.cpu().numpy())
        return np.argmin(distance, axis=0), aux_dataloader


class ClassVarQuery(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model

    def compute_score(self, model_output, image):
        model_output, _, _ = model_output
        if len(model_output) == 1:
            o = model_output[0]
        elif len(model_output) > 1:
            o = torch.stack(model_output).mean(0)
        else:
            raise NotImplementedError
        return f.class_var_score(o, image)

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        device = next(iter(self.model.parameters())).device
        q = LimitSortedList(limit=query_num, descending=False)
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(device)
            output = self.model(img)
            score = self.compute_score(output, img).cpu()
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack(
                [torch.arange(offset, offset + len(img)), score])
            q.extend(idx_entropy)
        return q.data


class ClassVarVarQuery(ClassVarQuery):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = int(kwargs.get("alpha", 2))
        self.dataset = self.unlabeled_dataloader.dataset
        self.device = next(iter(self.model.parameters())).device

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        qn = query_num * self.alpha
        dataset_idx = super().select_dataset_idx(qn)
        img_idx = self.convert2img_idx(dataset_idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        q = LimitSortedList(limit=query_num, descending=True)
        self.model.eval()
        for batch_idx, (img, _) in enumerate(aux_dataloader):
            img = img.to(self.device)
            pred, _, _ = self.model(img)
            score = f.var(torch.stack(pred))
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * aux_dataloader.batch_size
            idx_score = torch.column_stack(
                [torch.arange(offset, offset + len(img)),
                 score.cpu()])
            q.extend(idx_score)
        return q.data, aux_dataloader


class KMeansVarUnionQuery(AverageAlignConsistency):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        pool_size = int(kwargs.get("pool_size", 12))
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((pool_size, pool_size)))
        self.pool.eval()
        self.cycle = 1

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        num_var = query_num * self.cycle
        num_kmeans = query_num * (4 - self.cycle)
        self.model.eval()

        q = LimitSortedList(limit=num_var, descending=self.descending)
        embedding_list = []
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(self.device)
            output = self.model(img)
            score = self.compute_score(output).cpu()
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack(
                [torch.arange(offset, offset + len(img)), score])
            q.extend(idx_entropy)

            features = output[2]
            embedding = self.pool(features[0]).view((img.shape[0], -1))
            embedding_list.append(embedding)
        varseleced = np.asanyarray(list(q.data))

        e = torch.concat(embedding_list, dim=0)
        mkmeans = KMeans(num_kmeans, init="k-means++", n_init="auto")
        distance = mkmeans.fit_transform(e.cpu().numpy())
        kmeans_seleced = np.unique(np.argmin(distance, axis=0))

        union = np.unique(np.concatenate([varseleced, kmeans_seleced]))
        self.cycle += 1
        np.random.shuffle(union)
        return union[:query_num]


class MDistance:

    def __init__(self) -> None:
        self.ec = EmpiricalCovariance(assume_centered=True)

    def fit(self, X):
        self.ec.fit(X)

    def distance(self, x):
        return self.ec.mahalanobis(x)


class FeatureClassVarianceQuery(SimpleQueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, descending=True, **kwargs)

    def compute_score(self, model_output):
        return f.inner_class_var_outer_class_div_feature(model_output)


# class GMClusterQuery(QueryStrategy):
#     def __init__(self, dataloader: DataLoader, **kwargs) -> None:
#         super().__init__(dataloader)
#         assert "trainer" in kwargs
#         self.trainer = kwargs["trainer"]
#         self.model = kwargs["trainer"].model
#         self.device = next(iter(self.model.parameters())).device

#         pool_size = int(kwargs.get("pool_size", 12))
#         self.feature = self.build_feature_layer(pool_size=pool_size)

#     def build_feature_layer(self, pool_size):
#         d = 512 if self.trainer.model.ft_chns[0] == 32 else 256

#         pool = nn.AdaptiveAvgPool2d(
#             (pool_size, pool_size)).to(self.device)
#         con1x1 = nn.Conv2d(d, d,
#                            kernel_size=pool_size, bias=False).to(self.device)
#         relu = nn.ReLU(True)
#         bn = nn.BatchNorm2d(d).to(self.device)
#         feature = nn.Sequential(pool, con1x1, bn, relu)
#         feature.eval()
#         return feature

#     @torch.no_grad()
#     def embedding(self, dataloader):

#         embedding_list = []
#         device = next(iter(self.model.parameters())).device
#         for _, (img, _) in enumerate(dataloader):
#             img = img.to(device)
#             _, _, features = self.model(img)
#             embedding = self.feature(features[0]).flatten(
#                 1, -1).cpu().numpy()
#             embedding_list.append(embedding)
#         return np.concatenate(embedding_list, axis=0)

#     def select_dataset_idx(self, query_num):
#         self.model.eval()
#         features = self.embedding(self.unlabeled_dataloader)

#         ec = MDistance()
#         ec.fit(features)
#         distance = ec.distance(features)[:, None]
#         gmm = GaussianMixture(n_components=5*query_num,
#                               init_params="k-means++")
#         gmm.fit(distance)
#         prob = gmm.predict_proba(distance)
#         prob.sort(axis=1)
#         score = prob[:, -1] - prob[:, -2]

#         return np.unique(score.argsort())[-query_num:]

# class GMClusterVarUnionQuery(AverageAlignConsistency):
#     def __init__(self, dataloader: DataLoader, **kwargs) -> None:
#         super().__init__(dataloader, **kwargs)
#         pool_size = int(kwargs.get("pool_size", 12))
#         self.cycle = 1
#         self.feature = self.build_feature_layer(pool_size=pool_size)

#     def build_feature_layer(self, pool_size):
#         d = 512 if self.trainer.model.ft_chns[0] == 32 else 256

#         pool = nn.AdaptiveAvgPool2d(
#             (pool_size, pool_size)).to(self.device)
#         con1x1 = nn.Conv2d(d, d,
#                            kernel_size=pool_size, bias=False).to(self.device)
#         relu = nn.ReLU(True)
#         bn = nn.BatchNorm2d(d).to(self.device)
#         feature = nn.Sequential(pool, con1x1, bn, relu)
#         feature.eval()
#         return feature

#     @torch.no_grad()
#     def select_dataset_idx(self, query_num):
#         num_var = query_num*self.cycle
#         num_cluster = query_num*(4 - self.cycle)
#         self.model.eval()

#         q = LimitSortedList(limit=num_var, descending=self.descending)
#         embedding_list = []
#         for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
#             img = img.to(self.device)
#             output = self.model(img)
#             score = self.compute_score(output).cpu()
#             assert len(score) == len(img), "shape mismatch!"
#             offset = batch_idx * self.unlabeled_dataloader.batch_size
#             idx_entropy = torch.column_stack(
#                 [torch.arange(offset, offset + len(img)), score])
#             q.extend(idx_entropy)

#             features = output[2]
#             embedding = self.feature(features[0]).flatten(
#                 1, -1).cpu().numpy()
#             embedding_list.append(embedding)
#         varseleced = np.asanyarray(list(q.data))

#         features = np.concatenate(embedding_list)
#         ec = MDistance()
#         ec.fit(features)
#         distance = ec.distance(features)[:, None]
#         gmm = GaussianMixture(n_components=5*num_cluster,
#                               init_params="k-means++")
#         gmm.fit(distance)
#         prob = gmm.predict_proba(distance)
#         prob.sort(axis=1)
#         score = prob[:, -1] - prob[:, -2]
#         cluster_selected = np.unique(score.argsort())[-query_num:]

#         union = np.unique(np.concatenate([varseleced, cluster_selected]))
#         self.cycle += 1
#         np.random.shuffle(union)
#         return union[:query_num]

# class GMClusterVarIntersectionQuery(AverageAlignConsistency):
#     def __init__(self, dataloader: DataLoader, **kwargs) -> None:
#         super().__init__(dataloader, **kwargs)
#         pool_size = int(kwargs.get("pool_size", 12))
#         self.cycle = 1
#         self.feature = self.build_feature_layer(pool_size=pool_size)

#     def build_feature_layer(self, pool_size):
#         d = 512 if self.trainer.model.ft_chns[0] == 32 else 256

#         pool = nn.AdaptiveAvgPool2d(
#             (pool_size, pool_size)).to(self.device)
#         con1x1 = nn.Conv2d(d, d,
#                            kernel_size=pool_size, bias=False).to(self.device)
#         relu = nn.ReLU(True)
#         bn = nn.BatchNorm2d(d).to(self.device)
#         feature = nn.Sequential(pool, con1x1, bn, relu)
#         feature.eval()
#         return feature

#     def intersection(self, a, b, query_num):
#         selected = []
#         for i in a:
#             if i in b:
#                 selected.append(i)
#                 if len(selected) == query_num:
#                     break
#         return selected

#     def margin_confidence(score):
#         torch.sort(score, dim=0,)

#     @torch.no_grad()
#     def select_dataset_idx(self, query_num):
#         self.model.eval()

#         q = []
#         embedding_list = []
#         for _, (img, _) in enumerate(self.unlabeled_dataloader):
#             img = img.to(self.device)
#             output = self.model(img)
#             score = self.compute_score(output).cpu()
#             assert len(score) == len(img), "shape mismatch!"
#             q.extend(score)

#             features = output[2]
#             embedding = self.feature(features[0]).flatten(
#                 1, -1).cpu().numpy()
#             embedding_list.append(embedding)

#         var_sorted = np.asanyarray(q).argsort()[::-1]

#         features = np.concatenate(embedding_list)
#         ec = MDistance()
#         ec.fit(features)
#         distance = ec.distance(features)[:, None]
#         gmm = GaussianMixture(n_components=5*query_num,
#                               init_params="k-means++")
#         gmm.fit(distance)
#         prob = gmm.predict_proba(distance)
#         prob.sort(axis=1)
#         score = prob[:, -1] - prob[:, -2]
#         cluster_sorted = np.unique(score.argsort())

#         intersection = self.intersection(
#             cluster_sorted, var_sorted, query_num)
#         return intersection

# class GMClusterLabeledQuery(QueryStrategy):
#     def __init__(self, dataloader: DataLoader, **kwargs) -> None:
#         super().__init__(dataloader)
#         assert "trainer" in kwargs
#         self.trainer = kwargs["trainer"]
#         self.model = kwargs["trainer"].model
#         self.device = next(iter(self.model.parameters())).device

#         pool_size = int(kwargs.get("pool_size", 12))
#         self.feature = self.build_feature_layer(pool_size=pool_size)

#     def build_feature_layer(self, pool_size):
#         d = 512 if self.trainer.model.ft_chns[0] == 32 else 256

#         pool = nn.AdaptiveAvgPool2d(
#             (pool_size, pool_size)).to(self.device)
#         con1x1 = nn.Conv2d(d, d,
#                            kernel_size=pool_size, bias=False).to(self.device)
#         relu = nn.ReLU(True)
#         bn = nn.BatchNorm2d(d).to(self.device)
#         feature = nn.Sequential(pool, con1x1, bn, relu)
#         feature.eval()
#         return feature

#     @torch.no_grad()
#     def embedding(self, dataloader):

#         embedding_list = []
#         device = next(iter(self.model.parameters())).device
#         for _, (img, _) in enumerate(dataloader):
#             img = img.to(device)
#             _, _, features = self.model(img)
#             embedding = self.feature(features[0]).flatten(
#                 1, -1).cpu().numpy()
#             embedding_list.append(embedding)
#         return np.concatenate(embedding_list, axis=0)

#     def select_dataset_idx(self, query_num):
#         self.model.eval()
#         labeled_feature = self.embedding(self.labeled_dataloader)
#         unlabeled_features = self.embedding(self.unlabeled_dataloader)

#         ec = MDistance()
#         ec.fit(labeled_feature)

#         distance = ec.distance(unlabeled_features)[:, None]
#         gmm = GaussianMixture(n_components=5*query_num,
#                               init_params="k-means++")
#         gmm.fit(distance)
#         prob = gmm.predict_proba(distance)
#         prob.sort(axis=1)
#         score = prob[:, -1] - prob[:, -2]

#         return np.unique(score.argsort())[-query_num:]


class GMClusterQuery(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.device = next(iter(self.model.parameters())).device

        pool_size = int(kwargs.get("pool_size", 12))
        self.feature = self.build_feature_layer(pool_size=pool_size)

    def build_feature_layer(self, pool_size):
        d = 512 if self.trainer.model.ft_chns[0] == 32 else 256

        pool = nn.AdaptiveAvgPool2d((pool_size, pool_size)).to(self.device)
        con1x1 = nn.Conv2d(d, d, kernel_size=pool_size,
                           bias=False).to(self.device)
        relu = nn.ReLU(True)
        bn = nn.BatchNorm2d(d).to(self.device)
        feature = nn.Sequential(pool, con1x1, bn, relu)
        feature.eval()
        return feature

    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for _, (img, _) in enumerate(dataloader):
            img = img.to(device)
            _, _, features = self.model(img)
            embedding = self.feature(features[0]).flatten(1, -1).cpu().numpy()
            embedding_list.append(embedding)
        return np.concatenate(embedding_list, axis=0)

    # def collect_idx(self, data, query_num):
    #     sorted_list = sorted(data, key=lambda x: (x[1], -x[2]))
    #     lab = np.asarray(sorted_list)
    #     unique_cluster = np.unique(lab[:, 1])

    #     counter = {}
    #     print(sorted_list)
    #     for r in sorted_list:
    #         value = counter.get(r[1], [])
    #         value.append(int(r[0]))
    #         counter[int(r[1])] = value
    #     print(counter)
    #     sample_per_cluster = query_num//len(unique_cluster)
    #     q = []
    #     surplus = query_num - sample_per_cluster*len(unique_cluster)
    #     for l in list(counter.keys()):
    #         if len(counter[l]) <= sample_per_cluster:
    #             q += counter[l]
    #             surplus += sample_per_cluster - len(counter[l])
    #             del counter[l]
    #         else:
    #             q += counter[l][:sample_per_cluster]

    #     v = [].extend(counter.values())
    #     random.shuffle(v)
    #     q += v[:surplus]
    #     print(q)
    #     return q

    # def select_dataset_idx(self, query_num):
    #     self.model.eval()
    #     features = self.embedding(self.unlabeled_dataloader)

    #     gmm = GaussianMixture(n_components=query_num,
    #                           init_params="k-means++")
    #     gmm.fit(features)
    #     prob = gmm.predict_proba(features)
    #     prob.sort(1)
    #     score = prob[:, -1] - prob[:, -2]
    #     lab = prob.argmax(1)
    #     combine = np.array(np.column_stack(
    #         [range(len(lab)), lab, score]))
    #     return self.collect_idx(combine, query_num)
    def select_dataset_idx(self, query_num):
        self.model.eval()
        features = self.embedding(self.unlabeled_dataloader)

        gmm = GaussianMixture(n_components=5 * query_num,
                              init_params="k-means++")
        gmm.fit(features)
        labels = gmm.predict(features)
        counter = {}
        for i, l in enumerate(labels):
            value = counter.get(l, [])
            value.append(i)
            counter[l] = value

        size = len(features)
        q = []
        candidate = []
        for l in list(counter.keys()):
            sample_ratio = int(len(counter[l]) * query_num / size)
            if sample_ratio > 0:
                random.shuffle(counter[l])
                q += counter[l][:sample_ratio]
                counter[l] = counter[l][sample_ratio:]
            else:
                candidate += counter[l]

        surplus = query_num - len(q)
        np.random.shuffle(candidate)
        q += candidate[:surplus]
        return np.asanyarray(q)


class GMClusterVarUnionQuery(AverageAlignConsistency):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        pool_size = int(kwargs.get("pool_size", 12))
        self.cycle = 1
        self.feature = self.build_feature_layer(pool_size=pool_size)

    def build_feature_layer(self, pool_size):
        d = 512 if self.trainer.model.ft_chns[0] == 32 else 256

        pool = nn.AdaptiveAvgPool2d((pool_size, pool_size)).to(self.device)
        con1x1 = nn.Conv2d(d, d, kernel_size=pool_size,
                           bias=False).to(self.device)
        relu = nn.ReLU(True)
        bn = nn.BatchNorm2d(d).to(self.device)
        feature = nn.Sequential(pool, con1x1, bn, relu)
        feature.eval()
        return feature

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        num_var = query_num * self.cycle
        num_cluster = query_num * (4 - self.cycle)
        self.model.eval()

        q = LimitSortedList(limit=num_var, descending=self.descending)
        embedding_list = []
        for batch_idx, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(self.device)
            output = self.model(img)
            score = self.compute_score(output).cpu()
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * self.unlabeled_dataloader.batch_size
            idx_entropy = torch.column_stack(
                [torch.arange(offset, offset + len(img)), score])
            q.extend(idx_entropy)

            features = output[2]
            embedding = self.feature(features[0]).flatten(1, -1).cpu().numpy()
            embedding_list.append(embedding)
        varseleced = np.asanyarray(list(q.data))

        features = np.concatenate(embedding_list)
        gmm = GaussianMixture(n_components=5 * num_cluster,
                              init_params="k-means++")
        gmm.fit(features)
        labels = gmm.predict(features)
        counter = {}
        for i, l in enumerate(labels):
            value = counter.get(l, [])
            value.append(i)
            counter[l] = value

        size = len(features)
        q = []
        candidate = []
        for l in list(counter.keys()):
            sample_ratio = int(len(counter[l]) * query_num / size)
            if sample_ratio > 0:
                random.shuffle(counter[l])
                q += counter[l][:sample_ratio]
                counter[l] = counter[l][sample_ratio:]
            else:
                candidate += counter[l]

        surplus = query_num - len(q)
        np.random.shuffle(candidate)
        q += candidate[:surplus]
        cluster_selected = np.asanyarray(q)

        union = np.unique(np.concatenate([varseleced, cluster_selected]))
        self.cycle += 1
        np.random.shuffle(union)
        return union[:query_num]


class GMClusterVarIntersectionQuery(AverageAlignConsistency):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        pool_size = int(kwargs.get("pool_size", 12))
        self.cycle = 1
        self.feature = self.build_feature_layer(pool_size=pool_size)

    def build_feature_layer(self, pool_size):
        d = 512 if self.trainer.model.ft_chns[0] == 32 else 256

        pool = nn.AdaptiveAvgPool2d((pool_size, pool_size)).to(self.device)
        con1x1 = nn.Conv2d(d, d, kernel_size=pool_size,
                           bias=False).to(self.device)
        relu = nn.ReLU(True)
        bn = nn.BatchNorm2d(d).to(self.device)
        feature = nn.Sequential(pool, con1x1, bn, relu)
        feature.eval()
        return feature

    def intersection(self, a, b, query_num):
        selected = []
        for i in a:
            if i in b:
                selected.append(i)
                if len(selected) == query_num:
                    break
        return selected

    def margin_confidence(score):
        torch.sort(
            score,
            dim=0,
        )

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()

        q = []
        embedding_list = []
        for _, (img, _) in enumerate(self.unlabeled_dataloader):
            img = img.to(self.device)
            output = self.model(img)
            score = self.compute_score(output).cpu()
            assert len(score) == len(img), "shape mismatch!"
            q.extend(score)

            features = output[2]
            embedding = self.feature(features[0]).flatten(1, -1).cpu().numpy()
            embedding_list.append(embedding)

        var_sorted = np.asanyarray(q).argsort()[::-1]

        features = np.concatenate(embedding_list)
        gmm = GaussianMixture(n_components=5 * query_num,
                              init_params="k-means++")
        gmm.fit(features)
        prob = gmm.predict_proba(features)
        prob.sort(axis=1)
        score = prob[:, -1] - prob[:, -2]
        cluster_sorted = np.unique(score.argsort())

        intersection = self.intersection(cluster_sorted, var_sorted, query_num)
        return intersection


class GMClusterMLabeledQuery(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.device = next(iter(self.model.parameters())).device

        pool_size = int(kwargs.get("pool_size", 12))
        self.feature = self.build_feature_layer(pool_size=pool_size)

    def build_feature_layer(self, pool_size):
        d = 512 if self.trainer.model.ft_chns[0] == 32 else 256

        pool = nn.AdaptiveAvgPool2d((pool_size, pool_size)).to(self.device)
        con1x1 = nn.Conv2d(d, d, kernel_size=pool_size,
                           bias=False).to(self.device)
        relu = nn.ReLU(True)
        bn = nn.BatchNorm2d(d).to(self.device)
        feature = nn.Sequential(pool, con1x1, bn, relu)
        feature.eval()
        return feature

    @torch.no_grad()
    def embedding(self, dataloader):

        embedding_list = []
        device = next(iter(self.model.parameters())).device
        for _, (img, _) in enumerate(dataloader):
            img = img.to(device)
            _, _, features = self.model(img)
            embedding = self.feature(features[0]).flatten(1, -1).cpu().numpy()
            embedding_list.append(embedding)
        return np.concatenate(embedding_list, axis=0)

    def select_dataset_idx(self, query_num):
        self.model.eval()
        labeled_feature = self.embedding(self.labeled_dataloader)
        unlabeled_features = self.embedding(self.unlabeled_dataloader)

        gmm = GaussianMixture(n_components=5 * query_num,
                              init_params="k-means++")
        gmm.fit(labeled_feature)

        prob = gmm.predict_proba(unlabeled_features)
        prob.sort(axis=1)
        score = prob[:, -1] - prob[:, -2]

        return np.unique(score.argsort())[query_num:]


class VarGMClusterQuery(GMClusterQuery):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = int(kwargs.get("alpha", 2))
        self.dataset = self.unlabeled_dataloader.dataset
        self.device = next(iter(self.model.parameters())).device

    @torch.no_grad()
    def compute_score(self, model_output):
        model_output = torch.stack(model_output[0]).softmax(2)
        avg_pred = torch.mean(model_output, dim=0) * 0.99 + 0.005
        consistency = torch.zeros(len(model_output[1]), device=self.device)
        for aux in model_output:
            aux = aux * 0.99 + 0.005
            var = torch.sum(
                nn.functional.kl_div(aux.log(), avg_pred, reduction="none"),
                dim=1,
                keepdim=True,
            )
            exp_var = torch.exp(-var)
            square_e = torch.square(avg_pred - aux)
            c = torch.mean(square_e * exp_var, dim=[-1, -2, -3]) / (
                torch.mean(exp_var, dim=[-1, -2, -3]) + 1e-8)
            consistency += c
        return consistency

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        qn = query_num * self.alpha
        dataset_idx = super().select_dataset_idx(qn)
        img_idx = self.convert2img_idx(dataset_idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        q = LimitSortedList(limit=query_num, descending=True)
        self.model.eval()
        for batch_idx, (img, _) in enumerate(aux_dataloader):
            img = img.to(self.device)
            model_output = self.model(img)
            score = self.compute_score(model_output)
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * aux_dataloader.batch_size
            idx_score = torch.column_stack(
                [torch.arange(offset, offset + len(img)),
                 score.cpu()])
            q.extend(idx_score)
        return q.data, aux_dataloader


class GMClusterVarQuery(AverageAlignConsistency):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = int(kwargs.get("alpha", 2))
        self.dataset = self.unlabeled_dataloader.dataset
        self.device = next(iter(self.model.parameters())).device
        pool_size = int(kwargs.get("pool_size", 12))
        self.feature = self.build_feature_layer(pool_size=pool_size)

    def build_feature_layer(self, pool_size):
        d = 512 if self.trainer.model.ft_chns[0] == 32 else 256

        pool = nn.AdaptiveAvgPool2d((pool_size, pool_size)).to(self.device)
        con1x1 = nn.Conv2d(d, d, kernel_size=pool_size,
                           bias=False).to(self.device)
        relu = nn.ReLU(True)
        bn = nn.BatchNorm2d(d).to(self.device)
        feature = nn.Sequential(pool, con1x1, bn, relu)
        feature.eval()
        return feature

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        qn = query_num * self.alpha
        dataset_idx = super().select_dataset_idx(qn)
        img_idx = self.convert2img_idx(dataset_idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        self.model.eval()
        embedding_list = []
        for _, (img, _) in enumerate(aux_dataloader):
            img = img.to(self.device)
            model_output = self.model(img)
            features = model_output[2]
            embedding = self.feature(features[0]).flatten(1, -1).cpu().numpy()
            embedding_list.append(embedding)

        features = np.concatenate(embedding_list)
        gmm = GaussianMixture(n_components=5 * query_num,
                              init_params="k-means++")
        gmm.fit(features)
        labels = gmm.predict(features)
        counter = {}
        for i, l in enumerate(labels):
            value = counter.get(l, [])
            value.append(i)
            counter[l] = value

        size = len(features)
        q = []
        candidate = []
        for l in list(counter.keys()):
            sample_ratio = int(len(counter[l]) * query_num / size)
            if sample_ratio > 0:
                random.shuffle(counter[l])
                q += counter[l][:sample_ratio]
                counter[l] = counter[l][sample_ratio:]
            else:
                candidate += counter[l]

        surplus = query_num - len(q)
        np.random.shuffle(candidate)
        q += candidate[:surplus]
        cluster_selected = np.asanyarray(q)

        return cluster_selected, aux_dataloader


class MahalanobisDistance(QueryStrategy):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        print(kwargs)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.measure = kwargs.get("measure", "cosine")
        print(self.measure)
        self.device = next(iter(self.model.parameters())).device
        self.input_size = self.trainer.config["Dataset"]["input_size"]
        self.input_size = (np.array(self.input_size)
                           if type(self.input_size) == list else np.array(
                               [self.input_size, self.input_size]))
        self.d = 512 if self.trainer.model.ft_chns[0] == 32 else 256
        print(kwargs.get("uncertainty_measure", "entropy"))
        self.uncertainty_score = self.build_uncertainty(
            kwargs.get("uncertainty_measure", "entropy"))
        print(self.uncertainty_score)

    def build_uncertainty(self, uncertainty_measure):
        if uncertainty_measure == "entropy":
            return lambda x: -f.max_entropy(x)
        elif uncertainty_measure == "least_confidence":
            return f.least_confidence
        elif uncertainty_measure == "margin_confidence":
            return f.margin_confidence
        else:
            raise NotImplementedError

    def get_pixel_lab(self, mask):
        kernel_size = self.input_size / 32

        mask = mask.cpu().numpy()
        arr = []

        for m in mask:
            arr.append(mode_filter(m[0], kernel_size).flatten())
        return np.concatenate(arr)

    def group_by_class(self, embedding, label):
        class_feature = {}
        for f, l in zip(embedding, label):
            class_feature.setdefault(int(l), []).append(f)
        return class_feature

    @torch.no_grad()
    def labeled_embedding(self, dataloader):
        d = 512 if self.trainer.model.ft_chns[0] == 32 else 256
        device = next(iter(self.model.parameters())).device
        feature_list = []
        pix_mask_list = []
        for _, (img, mask) in enumerate(dataloader):
            img = img.to(device)
            pix_mask = self.get_pixel_lab(mask)
            pix_mask_list.append(pix_mask)
            _, _, features = self.model(img)
            feature = features[0]

            flatten_feature = feature.permute(0, 2, 3, 1).reshape(-1, d)
            feature_list.append(flatten_feature.cpu().numpy())

        return np.concatenate(feature_list), np.concatenate(pix_mask_list)

    @torch.no_grad()
    def unlabeled_embedding(self, dataloader):

        device = next(iter(self.model.parameters())).device

        feature_list = []
        for _, (img, _) in enumerate(dataloader):
            img = img.to(device)
            _, _, features = self.model(img)
            feature = features[0]

            flatten_feature = feature.permute(0, 2, 3, 1).reshape(
                feature.shape[0], -1, self.d)

            feature_list.append(flatten_feature.cpu().numpy())
        return np.concatenate(feature_list)

    def prototypes(self, grouped_feature):
        class_center = []
        for k in grouped_feature.keys():
            class_center.append(np.stack(grouped_feature[k]).mean(0))
        return np.stack(class_center)

    def sim_score(self, group_feature, unlabeled_dataloader):
        class_center = self.prototypes(group_feature).transpose(1, 0)
        unlab_feature = self.unlabeled_embedding(unlabeled_dataloader)
        unlab_feature, class_center = unlab_feature / np.linalg.norm(
            unlab_feature, axis=2,
            keepdims=True), class_center / np.linalg.norm(
                class_center, axis=0, keepdims=True)
        feature_sim = np.matmul(unlab_feature, class_center)
        patch_wise_uncertainty = feature_sim.reshape(
            feature_sim.shape[0],
            int(self.input_size[0] / 8),
            int(self.input_size[1] / 8),
            feature_sim.shape[2],
        ).transpose(0, 3, 1, 2)
        patch_wise_uncertainty = torch.tensor(patch_wise_uncertainty,
                                              dtype=torch.float32,
                                              device=self.device).softmax(1)
        return self.uncertainty_score(patch_wise_uncertainty).cpu().numpy()

    def fit_model(self, group_feature):
        dis_calc = {}
        for k in group_feature.keys():
            class_features = np.stack(group_feature[k])
            dis_calc[k] = MDistance()
            dis_calc[k].fit(class_features)
        return dis_calc

    def distance_score(self, group_feature, unlabeled_dataloader):
        dis_calc = self.fit_model(group_feature)
        unlab_embedding = self.unlabeled_embedding(unlabeled_dataloader)
        numsamples = unlab_embedding.shape[0]
        flatten_embedding = unlab_embedding.reshape(-1, self.d)
        class_distance_list = []
        for d in dis_calc.keys():
            distance = dis_calc[d].distance(flatten_embedding)
            class_distance_list.append(distance)
        distance_score = (np.stack(class_distance_list, axis=1).reshape(
            numsamples,
            int(self.input_size[0] / 8),
            int(self.input_size[0] / 8),
            len(dis_calc.keys()),
        ).transpose(0, 3, 1, 2))

        patch_wise_uncertainty = torch.tensor(-distance_score,
                                              dtype=torch.float32,
                                              device=self.device).softmax(1)

        return self.uncertainty_score(patch_wise_uncertainty).cpu().numpy()

    def select_dataset_idx(self, query_num):
        pixel_feature, pixel_lab = self.labeled_embedding(
            self.labeled_dataloader)
        group_feature = self.group_by_class(pixel_feature, pixel_lab)
        if self.measure == "cosine":
            distance = self.sim_score(group_feature, self.unlabeled_dataloader)
        elif self.measure == "mahalanobis":
            distance = self.distance_score(group_feature,
                                           self.unlabeled_dataloader)
        else:
            raise NotImplementedError
        return distance.argsort()[:query_num]

        # group_feature = self.group_by_class(pixel_feature, pixel_lab)

        # class_center = self.prototypes(group_feature)


class MahalanobisDistanceVar(MahalanobisDistance):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = kwargs.get("alpha", 5)
        self.dataset = self.unlabeled_dataloader.dataset

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        feature_query = query_num * self.alpha
        idx = super().select_dataset_idx(feature_query)
        img_idx = self.convert2img_idx(idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        q = LimitSortedList(limit=query_num, descending=True)
        self.model.eval()
        for batch_idx, (img, _) in enumerate(aux_dataloader):
            img = img.to(self.device)
            pred, _, _ = self.model(img)
            score = f.var(torch.stack(pred))
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * aux_dataloader.batch_size
            idx_score = torch.column_stack(
                [torch.arange(offset, offset + len(img)), score.cpu()]
            )
            q.extend(idx_score)
        return q.data, aux_dataloader


class FeatureVar(MahalanobisDistance):

    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)

    def prototypes(self, grouped_feature):
        class_center = []
        for k in grouped_feature.keys():
            feature = np.stack(grouped_feature[k])
            gfeature = np.split(feature, 4, axis=-1)
            class_center.append(np.stack(gfeature).mean(1))
        return np.stack(class_center, axis=-1)

    def sim_score(self, group_feature, unlabeled_dataloader):
        class_center = self.prototypes(group_feature)
        unlab_feature = self.unlabeled_embedding(
            unlabeled_dataloader)
        split_feature = np.split(unlab_feature, 4, axis=-1)

        group_sim = []
        for center, feature in zip(class_center, split_feature):

            feature, center = feature / np.linalg.norm(
                feature, axis=2, keepdims=True
            ), center / np.linalg.norm(center, axis=0, keepdims=True)
            feature_sim = np.matmul(feature, center)
            patch_feature_sim = feature_sim.reshape(
                feature_sim.shape[0],
                int(self.input_size[0] / 8),
                int(self.input_size[1] / 8),
                feature_sim.shape[2],
            ).transpose(0, 3, 1, 2)
            group_sim.append(patch_feature_sim)
        socre = -f.var(torch.tensor(np.stack(group_sim),
                       device=self.device)).cpu().numpy()
        return socre

    def distance_score(self, group_feature, unlabeled_dataloader):
        dis_calc = self.fit_model(group_feature)
        unlab_embedding = self.unlabeled_embedding(unlabeled_dataloader)
        numsamples = unlab_embedding.shape[0]
        flatten_embedding = unlab_embedding.reshape(-1, self.d)
        class_distance_list = []
        for d in dis_calc.keys():
            group_list = []
            split_embedding = np.split(flatten_embedding, 4, axis=-1)
            for c, e in zip(dis_calc[d], split_embedding):
                distance = c.distance(e)
                group_list.append(distance)
            class_distance_list.append(np.stack(group_list))
        keylist = list(dis_calc.keys())
        score = np.stack(class_distance_list).reshape(len(keylist), len(dis_calc[keylist[0]]), numsamples, int(self.input_size[0] / 8),
                                                      int(self.input_size[0] / 8)).transpose(1, 2, 0, 3, 4)
        patch_wise_uncertainty = torch.tensor(
            -score, dtype=torch.float32, device=self.device
        )
        return -f.var(patch_wise_uncertainty).cpu().numpy()

    def fit_model(self, group_feature):
        dis_calc = {}
        for k in group_feature.keys():
            class_f = np.stack(group_feature[k])
            gfeatrues = np.split(class_f, 4, axis=-1)

            for f in gfeatrues:
                dis_measure = MDistance()
                dis_measure.fit(f)
                dis_calc.setdefault(k, []).append(dis_measure)
        return dis_calc

    def select_dataset_idx(self, query_num):
        pixel_feature, pixel_lab = self.labeled_embedding(
            self.labeled_dataloader)
        group_feature = self.group_by_class(pixel_feature, pixel_lab)
        if self.measure == "cosine":
            distance = self.sim_score(group_feature, self.unlabeled_dataloader)
        elif self.measure == "mahalanobis":
            distance = self.distance_score(
                group_feature, self.unlabeled_dataloader)
        else:
            raise NotImplementedError
        print(distance[:30])
        return distance.argsort()[:query_num]


class FeatureVarPredictionVar(FeatureVar):
    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = kwargs.get("alpha", 5)
        self.dataset = self.unlabeled_dataloader.dataset

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        feature_query = query_num * self.alpha
        idx = super().select_dataset_idx(feature_query)
        img_idx = self.convert2img_idx(idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        q = LimitSortedList(limit=query_num, descending=True)
        self.model.eval()
        for batch_idx, (img, _) in enumerate(aux_dataloader):
            img = img.to(self.device)
            pred, _, _ = self.model(img)
            score = f.var(torch.stack(pred))
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * aux_dataloader.batch_size
            idx_score = torch.column_stack(
                [torch.arange(offset, offset + len(img)), score.cpu()]
            )
            q.extend(idx_score)
        return q.data, aux_dataloader


class ClassFeatureUncertainty(QueryStrategy):
    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.device = next(iter(self.model.parameters())).device
        self.class_num = self.trainer.config['Network']['class_num']
        print(kwargs.get("uncertainty_measure", "entropy"))
        self.uncertainty_score = self.build_uncertainty(
            kwargs.get("uncertainty_measure", "entropy"))
        print(self.uncertainty_score)

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        class_bank = [[] for i in range(self.class_num)]
        for image, label in self.labeled_dataloader:
            image, label = image.to(self.device), label.to(self.device)
            _, _, feature = self.model(image)
            feature = feature[0]
            zoomed_label = F.interpolate(
                label.float(), feature.shape[2:], mode='nearest')
            feature = feature.permute(0, 2, 3, 1)
            for c in range(self.class_num):
                class_label = (zoomed_label == c).squeeze(1)
                class_bank[c].append(feature[class_label, :])

        for f in range(len(class_bank)):
            if len(class_bank[f]) == 0:
                class_bank[f] = torch.zeros(
                    class_bank[0].shape, dtype=torch.float, device=self.device)
            else:
                class_bank[f] = torch.cat(class_bank[f], dim=0).mean(0)

        gc.collect()
        class_center = torch.stack(class_bank, dim=1)
        # todo:统一一下都用np.argsort,不要limitsortedquery
        s = []
        for image, _ in self.unlabeled_dataloader:
            image = image.to(self.device)
            _, _, feature = self.model(image)
            feature = feature[0].permute(0, 2, 3, 1)
            patch_prediction = F.cosine_similarity(
                feature[..., None], class_center, dim=-2)
            s += self.uncertainty_score(
                patch_prediction.permute(0, 3, 1, 2)).cpu().numpy().tolist()
        q = np.argsort(np.asarray(s))[:query_num]
        print(q[:10])
        print(s[:10])
        return q

    def build_uncertainty(self, uncertainty_measure):
        # todo：应该是静态的
        if uncertainty_measure == "entropy":
            return lambda x: -f.max_entropy(x)
        elif uncertainty_measure == "least_confidence":
            return f.least_confidence
        elif uncertainty_measure == "margin_confidence":
            return f.margin_confidence
        else:
            raise NotImplementedError


class PseudoFeatureDistance(QueryStrategy):
    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader)
        assert "trainer" in kwargs
        self.trainer = kwargs["trainer"]
        self.model = kwargs["trainer"].model
        self.device = next(iter(self.model.parameters())).device
        self.class_num = self.trainer.config['Network']['class_num']

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        self.model.eval()
        class_bank = [[] for i in range(self.class_num)]
        for image, label in self.labeled_dataloader:
            image, label = image.to(self.device), label.to(self.device)
            _, _, feature = self.model(image)
            feature = feature[0]
            zoomed_label = F.interpolate(
                label.float(), feature.shape[2:], mode='nearest')
            feature = feature.permute(0, 2, 3, 1)
            for c in range(self.class_num):
                class_label = (zoomed_label == c).squeeze(1)
                class_bank[c].append(feature[class_label, :])

        for f in range(len(class_bank)):
            if len(class_bank[f]) == 0:
                class_bank[f] = torch.zeros(
                    class_bank[0].shape, dtype=torch.float, device=self.device)
            else:
                class_bank[f] = torch.cat(class_bank[f], dim=0).mean(0)

        gc.collect()
        class_center = torch.stack(class_bank, dim=1)
        # todo:统一一下都用np.argsort,不要limitsortedquery
        score = []
        for image, _ in self.unlabeled_dataloader:
            image = image.to(self.device)
            pred, _, feature = self.model(image)
            feature = feature[0]
            pred_mask = torch.stack(pred).mean(0).argmax(1, keepdim=True)
            pred_mask = F.interpolate(
                pred_mask.float(), feature.shape[2:], mode='nearest')

            feature = feature.permute(0, 2, 3, 1)
            for p, f in zip(pred_mask, feature):
                class_pos = torch.unique(p).long()
                distance = 0
                for c in class_pos:
                    pred_cls = (c == p).squeeze()
                    feature_c = f[pred_cls, :]
                    distance += F.cosine_similarity(feature_c,
                                                    class_center[:, c][None]).mean()
                score.append(distance.cpu().detach())
        q = np.argsort(np.asarray(score))[:query_num]
        return q


class PseudoFeatureDistanceVar(PseudoFeatureDistance):
    def __init__(self, dataloader: DataLoader, **kwargs) -> None:
        super().__init__(dataloader, **kwargs)
        self.alpha = kwargs.get("alpha", 5)
        self.dataset = self.unlabeled_dataloader.dataset

    @torch.no_grad()
    def select_dataset_idx(self, query_num):
        feature_query = query_num * self.alpha
        idx = super().select_dataset_idx(feature_query)
        img_idx = self.convert2img_idx(idx, self.unlabeled_dataloader)
        aux_dataloader = DataLoader(
            self.dataset,
            batch_size=self.unlabeled_dataloader.batch_size,
            sampler=SubsetSampler(img_idx),
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=4,
            num_workers=self.unlabeled_dataloader.num_workers,
        )
        q = LimitSortedList(limit=query_num, descending=True)
        self.model.eval()
        for batch_idx, (img, _) in enumerate(aux_dataloader):
            img = img.to(self.device)
            pred, _, _ = self.model(img)
            score = f.var(torch.stack(pred))
            assert len(score) == len(img), "shape mismatch!"
            offset = batch_idx * aux_dataloader.batch_size
            idx_score = torch.column_stack(
                [torch.arange(offset, offset + len(img)), score.cpu()]
            )
            q.extend(idx_score)

        return q.data, aux_dataloader
