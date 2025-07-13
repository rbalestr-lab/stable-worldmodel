import minari
import stable_ssl


def test_probing(num_steps=2):
    import lightning as pl
    import torch
    import torchmetrics
    from transformers import AutoConfig, AutoModelForImageClassification
    import torchvision
    import stable_ssl as ossl
    from stable_ssl.data import transforms

    # without transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        transforms.ToImage(
            mean=mean,
            std=std,
            source="observations.pixels",
            target="observations.pixels",
        ),
    )
    minari_dataset = minari.load_dataset("xenoworlds/ImagePositioning-v1")
    dataset = stable_ssl.data.MinariStepsDataset(
        minari_dataset, num_steps=num_steps, transform=transform
    )
    train_dataset, val_dataset = stable_ssl.data.random_split(
        dataset, lengths=[0.5, 0.5]
    )
    train = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=2, num_workers=20, drop_last=True
    )
    val = torch.utils.data.DataLoader(val_dataset, batch_size=2, num_workers=10)
    data = ossl.data.DataModule(train=train, val=val)

    def forward(self, batch, stage):
        if type(batch["actions"] is dict):
            actions = []
            for v in batch["actions"].values():
                actions.append(v.flatten(2))
            actions = torch.cat(actions, -1)
        else:
            actions = batch["actions"].flatten(2)
        batch["embedding"] = self.backbone(
            batch["observations"]["pixels"].flatten(0, 1)
        )["logits"]
        batch["prediction"] = self.predictor(
            torch.cat([batch["embedding"], actions.flatten(0, 1)], 1)
        )
        if self.training:
            past_embed = batch["embedding"].reshape(
                batch["observations"]["pixels"].shape[:2] + (512,)
            )[:, :-1]
            preds = batch["prediction"].reshape(
                batch["observations"]["pixels"].shape[:2] + (512,)
            )[:, 1:]
            batch["loss"] = torch.nn.functional.mse_loss(preds, past_embed)
        return batch

    config = AutoConfig.from_pretrained("microsoft/resnet-18")
    backbone = AutoModelForImageClassification.from_config(config)
    if type(dataset[0]["actions"]) is dict:
        action_dim = 0
        for v in dataset[0]["actions"].values():
            action_dim += v.size
    else:
        action_dim = dataset[0]["actions"].size
    action_dim //= num_steps
    predictor = torchvision.ops.MLP(
        512 + action_dim,
        [1024, 1024, 512],
        norm_layer=torch.nn.BatchNorm1d,
    )
    backbone.classifier[1] = torch.nn.Identity()
    module = ossl.Module(
        backbone=stable_ssl.backbone.EvalOnly(backbone),
        predictor=predictor,
        forward=forward,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        num_sanity_val_steps=1,
        precision="16-mixed",
        logger=False,
        enable_checkpointing=False,
    )
    manager = ossl.Manager(trainer=trainer, module=module, data=data)
    manager()


if __name__ == "__main__":
    test_probing()
