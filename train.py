model = PixelCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for images, _ in dataloader:  # images: (B, 1, 28, 28)
    images = images * 255
    images = images.long()  # pixel value: 0~255
    inputs = images.float() / 255.0  # 归一化输入
    logits = model(inputs)  # (B, 256, H, W)
    loss = criterion(logits, images.squeeze(1))  # target: (B, H, W)
    loss.backward()
    optimizer.step()



