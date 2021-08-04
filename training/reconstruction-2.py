encoder = VGG19Encoder()
decoder = Decoder()
optimizer = optim.Adam(decoder.parameters(), lr=0.0001)

StyleTransferDataset(self, "./datasets/train2017/", coco_annotations, wiki_path)
while True:
    train_epoch_reconstruct(encoder, decoder,



def train_epoch_reconstruct(encoder, decoder, dataloader, optimizer, epoch_num, writer, run):
    encoder.train()
    decoder.train()
    total_loss = 0
    for i, content_image in tqdm.tqdm(enumerate(dataloader),\
                                      total = len(dataloader),\
                                      dynamic_ncols = True):
        content_image = content_image.to(DEVICE)

        optimizer.zero_grad()
        reconstruction = decoder(encoder(content_image)[-1])

        if i % 300 == 0:
            show_tensor(reconstruction[0].detach().clone(),\
                        epoch_num, run, info = "recon1")
            show_tensor(content_image[0].detach().clone(),\
                        epoch_num, run, info = "orgnl1")
            show_tensor(reconstruction[1].detach().clone(),\
                        epoch_num, run, info = "recon2")
            show_tensor(content_image[1].detach().clone(),\
                        epoch_num, run, info = "orgnl2")

        loss = F.mse_loss(content_image, reconstruction)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        writer.add_scalar('Loss/train_it', loss.item(), epoch_num)

    writer.add_scalar('Loss/train', total_loss, epoch_num)
    print(f"Epoch {epoch_num}, Loss {total_loss}")
    return total_loss
