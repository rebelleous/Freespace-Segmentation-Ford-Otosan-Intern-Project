from model import UNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 4
epochs = 20
cuda = True
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'ex_images')
###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = UNet(n_channels=3, n_classes=2)

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()


# TRAINING THE NEURAL NETWORK
for epoch in tqdm.tqdm(range(epochs)):

    running_loss = 0
    #her epochda image'leri sırayla vermemek için random bir şekilde image'ler ve mask'ler karıştırılır.
    pair_IM=list(zip(train_input_path_list,train_label_path_list))
    np.random.shuffle(pair_IM)
    unzipped_object=zip(*pair_IM)
    zipped_list=list(unzipped_object)
    train_input_path_list=list(zipped_list[0])
    train_label_path_list=list(zipped_list[1])
    
    for ind in range(steps_per_epoch):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        #ilk girişte train_input_path_list[0:4] ilk 4 elemanı alır
        #ikinci döngüde train_input_list[4:8] ikinci 4 elemanı alır 
        #her seferinde batch_size kadar eleman ilerler
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)#fonksiyonlar parametreleri girilerek değişkene atandı 
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)
        #preprocess kısmındaki modele sokucamız verilerimiz parametreler girilerek hazırlandı 
        
        optimizer.zero_grad()#gradyanı sıfırlar yoksa her yinelemede birikme oluşur
        # Weights güncelledikten sonra gradientleri manuel olarak sıfırlayın

        outputs = model(batch_input) # modele batch_inputu parametre olarak verip oluşan çıktıyı değişkene atadık 
        

        # Forward passes the input data
        loss = criterion(outputs, batch_label)#hedef ve çıktı arasındaki ikili çapraz entropiyi ölçer 
        loss.backward()# Gradyanı hesaplar, her bir parametrenin ne kadar güncellenmesi gerektiğini verir
        optimizer.step()# Gradyana göre her parametreyi günceller

        running_loss += loss.item()# loss.item (), loss'da tutulan skaler değeri alır.

        print(ind)
        #validation 
        if ind == steps_per_epoch-1:
            
            train_losses.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)# modele batch_inputu parametre olarak verip oluşan çıktıyı değişkene atadık 
                loss = criterion(outputs, batch_label)#hedef ve çıktı arasındaki ikili çapraz entropiyi ölçer 
                val_loss += loss.item()
                val_losses.append(val_loss)
                break

            print('validation loss on epoch {}: {}'.format(epoch, val_loss))
            
torch.save(outputs, '/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/src/best_model.pth')
print("Model Saved!")
best_model = torch.load('/home/aycaburcu/Masaüstü/Ford_Otosan_Intern/src/best_model.pth')

def draw_graph(val_losses,train_losses,epochs):
    norm_validation = [float(i)/sum(val_losses) for i in val_losses]
    norm_train = [float(i)/sum(train_losses) for i in train_losses]
    epoch_numbers=list(range(1,epochs+1,1))
    plt.figure(figsize=(12,6))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_numbers,norm_validation,color="red") 
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Train losses')
    plt.subplot(2, 2, 2)
    plt.plot(epoch_numbers,norm_train,color="blue")
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    plt.title('Validation losses')
    plt.subplot(2, 1, 2)
    plt.plot(epoch_numbers,norm_validation, 'r-',color="red")
    plt.plot(epoch_numbers,norm_train, 'r-',color="blue")
    plt.legend(['w=1','w=2'])
    plt.title('Train and Validation Losses')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
    
    
    plt.show()

draw_graph(val_losses,train_losses,epochs)


def predict(test_input_path_list):

    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out=torch.argmax(outs,axis=1)
        out_cpu = out.cpu()
        outputs_list=out_cpu.detach().numpy()
        mask=np.squeeze(outputs_list,axis=0)
            
            
        img=cv2.imread(batch_test[0])
        mg=cv2.resize(img,(224,224))
        mask_ind   = mask == 1
        cpy_img  = mg.copy()
        mg[mask==1 ,:] = (255, 0, 125)
        opac_image=(mg/2+cpy_img/2).astype(np.uint8)
        predict_name=batch_test[0]
        predict_path=predict_name.replace('image', 'predict')
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))

predict(test_input_path_list)

