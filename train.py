import torch.nn as nn
import torch.nn.functional as F


def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        images, captions = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        enc_memory_list = encoder(images)
        outputs,attn=decoder(tgt=captions,enc_memory_list=enc_memory_list)
        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per
            batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss



def Trainer(epochs,encoder,decoder,training_loader,validation_loader,loss_fn,optimizer):


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/TSTransformer_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = epochs
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        encoder.train(True)
        decoder.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        encoder.eval()
        decoder.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vlabels_idx=decoder.tokenizer(vlabels,max_length=decoder.max_seql,padding='max_length',return_tensors='pt')
                memory = encoder(vinputs)
                voutputs=decoder(tgt=vlabels,enc_memory_list=memory)
                vloss = loss_fn(voutputs, vlabels_idx)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            encoder_path = 'runs/enc_{}_{}'.format(timestamp, epoch_number)
            decoder_path= 'runs/dec_{}_{}'.format(timestamp, epoch_number)
            torch.save(encoder.state_dict(), encoder_path)
            torch.save(decoder.state_dict(), decoder_path)

        epoch_number += 1