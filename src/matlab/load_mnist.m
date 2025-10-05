clc, clear all
%% MNIST datapath
train_images_file = 'data/train-images.idx3-ubyte';
train_labels_file = 'data/train-labels.idx1-ubyte';
test_images_file  = 'data/t10k-images.idx3-ubyte';
test_labels_file  = 'data/t10k-labels.idx1-ubyte';

%% Load data
train_data = loadMNISTImages(train_images_file);
train_labels = loadMNISTLabels(train_labels_file);
test_data = loadMNISTImages(test_images_file);
test_labels = loadMNISTLabels(test_labels_file);

% Display dataset relevant information
fprintf('Train data length: %d images\n', length(train_data));
fprintf('Train labels length: %d labels\n', length(train_labels));
fprintf('Test data length: %d images\n', length(test_data));
fprintf('Test labels length: %d labels\n', length(test_labels));
fprintf('Train data shape: [%d %d %d]\n', size(train_data));
fprintf('Test data shape: [%d %d %d]\n', size(test_data));
fprintf('First label: %d\n', train_labels(1));
fprintf('Pixel value range: %d to %d\n', min(train_data(:)), max(train_data(:)));
fprintf('Unique labels: %s\n', mat2str(unique(train_labels)));

%% Reshape input and test data to 28x28
train_data = reshape(train_data, 28, 28, []);
test_data = reshape(train_data, 28, 28, []);

% Display new data shape
fprintf('Train data shape: [%d %d %d]\n', size(train_data));
fprintf('Test data shape: [%d %d %d]\n', size(test_data));
fprintf('Pixel value range: %d to %d\n', min(train_data(:)), max(train_data(:)));

%% Display 10 MNIST examples
figure;
for i = 1:10
    subplot(2,5,i);
    imshow(train_data(:,:,i), []);
    title(sprintf('Label: %d', train_labels(i)));
end
sgtitle('First 10 MNIST Training Images');