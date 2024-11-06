# we will do our training in main using our created dataloader
from dataset import

# paths for the datasets we created
train_color_folder = '.dataset/train/color'
train_grayscale_folder = '.dataset/train/grayscale'
test_color_folder = '.dataset/test/color'
test_grayscale_folder = '.dataset/test/grayscale'

# dataloaders being created for training and testing
train_dataset = create_Dataloader(train_color_folder,train_grayscale_folder)
test_dataset = create_Dataloader(test_color_folder,test_grayscale_folder)

# creating our model 
model = VideoColorizationCNN()

# Loss and optimizer, I am going to use the same one from CNN class example
# Mean Squared Error
criterion = nn.MSELoss()
# maybe change LR later, better to keep it smaller for now
optimizer = torch.optim.Adam(model.parameters(), lr=.001)

# step 4 on geeks for geeks is very important for backward propagation