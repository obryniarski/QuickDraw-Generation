from models import *
from data import *
from utils import *
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import time
import datetime
import os
import PIL.Image as Image

# device = (1 if torch.cuda.is_available() else 'cpu')

num_epochs = 100
batch_size = 128
learning_rate = 0.0002

lam = 0.01

real_label = 0.9
fake_label = 0

# print('Training for {} epochs with device {}:'.format(num_epochs, device))

def train(gen, disc, dataloader, model_name):

    G_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    D_opt = torch.optim.Adam(disc.parameters(), lr=learning_rate * 1, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    gen.train()
    disc.train()


    writer = SummaryWriter(log_dir='runs/' + model_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # writer.add_hparams({'lr': learning_rate, 'bsize': batch_size})

    num_iters = 1

    for epoch in range(num_epochs):
        print(' ----- Epoch {} ----- '.format(epoch))
        for batch in tqdm(dataloader):
            full_imgs, cut_imgs, classes = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            classes = classes.view(-1, 1).float()
            labels = torch.ones((len(full_imgs), 1)).to(device)

            # ----- Discriminator Training -----

            # real examples, with label 1
            D_opt.zero_grad()
            labels.fill_(real_label)
            real_predictions = disc(full_imgs, classes)
            real_Dloss = criterion(real_predictions, labels)
            real_Dloss.backward()
            real_acc = real_accuracy(real_predictions)

            # fake examples, with label 0
            labels.fill_(fake_label)
            z = torch.rand(size=(len(full_imgs), 100)).to(device)
            fake_imgs = gen(cut_imgs, classes, z)
            fake_predictions = disc(fake_imgs, classes)
            fake_Dloss = criterion(fake_predictions, labels)
            fake_Dloss.backward()
            fake_acc = fake_accuracy(fake_predictions)

            # fake_predictions = disc(full_imgs, (classes + 1) % 10)
            # fake_Dloss2 = criterion(fake_predictions, labels)
            # fake_Dloss2.backward()

            D_opt.step()
            disc_loss = fake_Dloss + real_Dloss

            # ----- Generator Training -----
            G_opt.zero_grad()
            labels.fill_(1)
            z = torch.rand(size=(len(full_imgs), 100)).to(device)
            fake_imgs = gen(cut_imgs, classes, z)
            gan_Gloss = criterion(disc(fake_imgs, classes), labels)

            # new_Gloss = base_loss(fake_imgs, cut_imgs)
            new_Gloss = 0

            Gloss = gan_Gloss + lam * new_Gloss
            Gloss.backward()
            G_opt.step()

            # writer.add_scalar('Loss/Generator', gan_Gloss, num_iters)
            writer.add_scalar('Loss/GeneratorTotal', Gloss, num_iters)
            writer.add_scalar('Loss/Discriminator', disc_loss, num_iters)
            writer.add_scalars('Accuracy', {'Real':real_acc, 'Fake':fake_acc}, num_iters)
            num_iters += 1


        # print('Disc Loss: {}  ----  Gen Loss: {}/{}'.format(disc_loss, new_loss, gen_loss))


        # checkpoint
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            root = 'models/' + model_name + '/'
            if not os.path.exists(root):
                os.mkdir(root)
            torch.save(gen.state_dict(), root + 'gen_' + str(epoch) + '.pt')
            torch.save(disc.state_dict(), root + 'disc_' + str(epoch) + '.pt')

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            fig = plot_fake_images(gen, cut_imgs, classes, 10)
            writer.add_figure('Generated Images', fig, global_step=epoch)


# https://github.com/Zeleni9/pytorch-wgan/blob/master/models/wgan_gradient_penalty.py
def get_infinite_batches(data_loader):
        while True:
            for i, (images, cutoffs, classes) in enumerate(data_loader):
                yield images.to(device), cutoffs.to(device), classes.to(device)

def train_wgan(gen, disc, dataloader, model_name):
    learning_rate = 0.0002
    G_opt = torch.optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    D_opt = torch.optim.Adam(disc.parameters(), lr=learning_rate * 1, betas=(0.5, 0.9))
    gen.train()
    disc.train()
    gp_lambda = 10.
    n_critic_runs = 5

    writer = SummaryWriter(log_dir='runs/' + model_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    # writer.add_hparams({'lr': learning_rate, 'bsize': batch_size})
    num_iters = 1

    data = get_infinite_batches(dataloader)

    for epoch in range(num_epochs):
        print(' ----- Epoch {} ----- '.format(epoch))

        for g_iter in tqdm(range(200)):

            for p in disc.parameters():
                p.requires_grad = True

            for _ in range(n_critic_runs):
                # ----- Discriminator Training -----
                one = torch.tensor(1, dtype=torch.float)
                mone = one * -1
                one = one.to(device)
                mone = mone.to(device)

                # real examples
                D_opt.zero_grad()
                full_imgs, cut_imgs, classes = next(data)
                classes = classes.view(-1, 1).float()

                real_predictions = disc(full_imgs, classes)
                # print('real_shape', real_predictions.shape)

                real_predictions = real_predictions.mean()
                # print(real_predictions)
                real_predictions.backward(mone)
                # real_acc = real_accuracy(real_predictions)

                # fake examples
                z = torch.rand(size=(len(full_imgs), 100)).to(device)
                fake_imgs = gen(cut_imgs, classes, z)
                fake_predictions = disc(fake_imgs, classes)
                # print('fake', fake_predictions.shape)
                fake_predictions = fake_predictions.mean()
                fake_predictions.backward(one)
                # fake_acc = fake_accuracy(fake_predictions)

                # fake_predictions = disc(full_imgs, (classes + 1) % 10)
                # fake_Dloss2 = criterion(fake_predictions, labels)
                # fake_Dloss2.backward()

                eps = torch.rand(len(full_imgs), 1, 1, 1).to(device)
                eps = eps.expand(full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], full_imgs.shape[3])
                x_hat = eps * full_imgs + (1 - eps) * fake_imgs
                x_hat = Variable(x_hat, requires_grad=True)
                disc_x_hat = disc(x_hat, classes)
                # disc_x_hat.backward(torch.ones(disc_x_hat.size()).to(device))
                # x_hat_grad = Variable(x_hat.grad, requires_grad=True)

                x_hat_grad = torch.autograd.grad(outputs=disc_x_hat, inputs=x_hat,
                                   grad_outputs=torch.ones(
                                       disc_x_hat.size()).to(device),
                                   create_graph=True, retain_graph=True)[0]
                x_hat_grad = x_hat_grad.view(x_hat_grad.size(0), -1)

                grad_penalty = ((x_hat_grad.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
                # print('penalty', grad_penalty.shape)
                grad_penalty.backward()

                critic_loss = real_predictions - fake_predictions
                # critic_loss.backward()
                D_opt.step()


            for p in disc.parameters():
                p.requires_grad = False  # to avoid computation

            # ----- Generator Training -----
            G_opt.zero_grad()
            z = torch.rand(size=(len(full_imgs), 100)).to(device)
            fake_imgs = gen(cut_imgs, classes, z)
            fake_predictions = disc(fake_imgs, classes)
            fake_predictions = fake_predictions.mean()
            fake_predictions.backward(mone)
            gan_Gloss = -fake_predictions
            # if epoch == 0:
            #     new_Gloss = base_loss(fake_imgs, cut_imgs) * lam
            #     new_Gloss.backward(one)
            # else:
            #     new_Gloss = base_loss(fake_imgs, cut_imgs) * lam
            #     new_Gloss.backward(one)
            # Gloss = gan_Gloss + lam * new_Gloss
            # Gloss.backward()
            G_opt.step()

            # gan_Gloss = -fake_predictions


            writer.add_scalar('Loss/Generator', gan_Gloss, num_iters)
            # writer.add_scalar('Loss/GeneratorTotal', new_Gloss, num_iters)
            writer.add_scalar('Loss/Critic', critic_loss, num_iters)
            writer.add_scalar('Extra/Gradient_Penalty', grad_penalty, num_iters)
            # writer.add_scalars('Accuracy', {'Real':real_acc, 'Fake':fake_acc}, num_iters)
            num_iters += 1


        # print('Disc Loss: {}  ----  Gen Loss: {}/{}'.format(disc_loss, new_loss, gen_loss))


        # checkpoint
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            root = 'models/' + model_name + '/'
            if not os.path.exists(root):
                os.mkdir(root)
            torch.save(gen.state_dict(), root + 'gen_' + str(epoch) + '.pt')
            torch.save(disc.state_dict(), root + 'disc_' + str(epoch) + '.pt')

def run():
    torch.multiprocessing.freeze_support()
    gen = Add_Skip_Generator().to(device)
    disc = Discriminator().to(device)
    transform = transforms.Compose([transforms.RandomAffine(degrees=20,
                                                    translate=(0.3, 0.3),
                                                       scale = (.8,1.1)),
                               transforms.ToTensor()])
    # transform = transforms.ToTensor()
    data = MNIST_traindata('data/', transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    train(gen, disc, dataloader, 'addition_of_input')

if __name__ == '__main__':
    run()
