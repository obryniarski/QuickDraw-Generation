from models import *
from data import *
from utils import *
from tqdm import tqdm
import time
import os
import PIL.Image as Image

device = (0 if torch.cuda.is_available() else 'cpu')

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

    disc_loss_arr = []
    gen_loss_arr = []
    new_loss_arr = []
    disc_realacc_arr = []
    disc_fakeacc_arr = []

    for epoch in range(num_epochs):
        num_iters = 0
        disc_loss = 0
        gen_loss = 0
        new_loss = 0
        real_acc = 0
        fake_acc = 0

        for batch in tqdm(dataloader):
            full_imgs, cut_imgs, class_imgs, classes = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
            classes = classes.view(-1, 1).float()
            labels = torch.ones((len(full_imgs), 1)).to(device)

            # ----- Discriminator Training -----

            # real examples, with label 1
            D_opt.zero_grad()
            labels.fill_(real_label)
            # real_inputs = torch.cat([full_imgs, class_imgs], dim=1)
            # print(classes, classes.shape)
            real_predictions = disc(full_imgs, classes)
            real_Dloss = criterion(real_predictions, labels)
            real_Dloss.backward()
            real_acc += real_accuracy(real_predictions)

            # fake examples, with label 0
            labels.fill_(fake_label)
            # imgs = torch.cat([cut_imgs, class_imgs], dim=1)
            z = torch.rand(size=(len(full_imgs), 100)).to(device)
            fake_imgs = gen(cut_imgs, classes, z)
            # fake_inputs = torch.cat([fake_imgs, class_imgs], dim=1)
            fake_predictions = disc(fake_imgs, classes)
            fake_Dloss = criterion(fake_predictions, labels)
            fake_Dloss.backward()
            fake_acc += fake_accuracy(fake_predictions)

            # fake_predictions = disc(full_imgs, (classes + 1) % 10)
            # fake_Dloss2 = criterion(fake_predictions, labels)
            # fake_Dloss2.backward()

            D_opt.step()

            disc_loss += fake_Dloss + real_Dloss

            # ----- Generator Training -----
            G_opt.zero_grad()
            labels.fill_(1)
            # imgs = torch.cat([cut_imgs, class_imgs], dim=1)
            z = torch.rand(size=(len(full_imgs), 100)).to(device)
            fake_imgs = gen(cut_imgs, classes, z)
            # fake_inputs = torch.cat([fake_imgs, class_imgs], dim=1)
            gan_Gloss = criterion(disc(fake_imgs, classes), labels)
            if epoch == 0:
                new_Gloss = base_loss(fake_imgs, cut_imgs)
            else:
                new_Gloss = base_loss(fake_imgs, cut_imgs)
            Gloss = gan_Gloss + lam * new_Gloss
            Gloss.backward()
            G_opt.step()
            gen_loss += gan_Gloss
            new_loss += new_Gloss

            num_iters += 1

        disc_loss /= num_iters
        gen_loss /= num_iters
        new_loss /= num_iters
        real_acc /= num_iters
        fake_acc /= num_iters

        disc_loss_arr.append(disc_loss)
        gen_loss_arr.append(gen_loss)
        new_loss_arr.append(new_loss)
        disc_realacc_arr.append(real_acc)
        disc_fakeacc_arr.append(fake_acc)

        print('Disc Loss: {}  ----  Gen Loss: {}/{}'.format(disc_loss,new_loss, gen_loss))


        # save loss graphs
        root = 'plots/' + model_name + '/'
        if not os.path.exists(root):
            os.mkdir(root)
        plt.plot(range(len(disc_loss_arr)), disc_loss_arr, label='disc')
        plt.plot(range(len(gen_loss_arr)), gen_loss_arr, label='gen')
        plt.legend()
        plt.title('GAN Loss')
        plt.savefig(root + 'loss.png')
        plt.clf()
        plt.plot(range(len(disc_realacc_arr)), disc_realacc_arr, label='real_acc')
        plt.plot(range(len(disc_fakeacc_arr)), disc_fakeacc_arr, label='fake_acc')
        plt.legend()
        plt.title('Discriminator Accuracy')
        plt.savefig(root + 'acc.png')
        plt.clf()

        # checkpoint
        if epoch % 1 == 0 or epoch == num_epochs - 1:
            root = 'models/' + model_name + '/'
            if not os.path.exists(root):
                os.mkdir(root)
            torch.save(gen.state_dict(), root + 'gen_' + str(epoch) + '.pt')
            torch.save(disc.state_dict(), root + 'disc_' + str(epoch) + '.pt')



def run():
    torch.multiprocessing.freeze_support()
    gen = Skip_Generator().to(device)
    disc = Discriminator().to(device)
    transform = transforms.Compose([transforms.RandomAffine(degrees=0,
                                                    translate=(0.2, 0.2),
                                                       scale = (.9,1.1)),
                               transforms.ToTensor()])
    # transform = transforms.ToTensor()
    data = MNIST_traindata('data/', transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    train(gen, disc, dataloader, 'dropout')

if __name__ == '__main__':
    run()
