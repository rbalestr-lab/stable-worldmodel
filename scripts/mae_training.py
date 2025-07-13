import minari
import stable_ssl
import torch

minari_dataset = minari.load_dataset("xenoworlds/ImagePositioning-v1")
dataset = stable_ssl.data.MinariStepsDataset(minari_dataset, num_steps=1)
print(len(dataset))
print(dataset[0]["rewards"])
dataset = stable_ssl.data.MinariStepsDataset(minari_dataset, num_steps=2)
print(len(dataset))
print(dataset[0]["rewards"])

print(dataset[0]["rewards"].shape)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=10
)
for data in loader:
    print(data["rewards"].shape)
    break
