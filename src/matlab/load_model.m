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

%% Reshape input and test data to 28x28
train_data = reshape(train_data, 28, 28, []);
test_data = reshape(test_data, 28, 28, []);

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


%% Create dataset
% Reshape [28, 28, 1, 60000] and single datatype
train_data_4D = reshape(train_data, 28, 28, 1, []);
train_data_4D = single(train_data_4D);
train_labels_cat = categorical(train_labels);
trainDataset = augmentedImageDatastore([28 28 1], train_data_4D, train_labels_cat);

%% Load model

net = importKerasNetwork('saved_models/mnist_cnn/model.h5');
% net = importNetworkFromTensorFlow('saved_models/mnist_fc/SavedModel');
% model = importKerasNetwork('saved_models/mnist_cnn/model.h5);

info = analyzeNetwork(net)

%% Modify NN 
% Classification output
correctClasses = string(0:9);


% Get original layers
layers = net.Layers;

newClassificationLayer = classificationLayer('Classes', categorical(correctClasses), 'Name', 'output');

% Replace output layer
layers(end) = newClassificationLayer;

% Assemply Network with modified output layer
net = assembleNetwork(layers);
info = analyzeNetwork(net)


%% Evaluate moded

% Test dataset
test_data = reshape(test_data, 28, 28, 1, []);
test_data = single(test_data);
test_labels_cat = categorical(test_labels);
testDataset = augmentedImageDatastore([28 28 1], test_data);

% Predict%% Evaluate moded

% Test dataset
test_data = reshape(test_data, 28, 28, 1, []);
test_data = single(test_data);
test_labels_cat = categorical(test_labels);
testDataset = augmentedImageDatastore([28 28 1], test_data);

% Predict

numSamples = size(test_data, 4);
tic;
predicted_labels = classify(net, test_data);
% predicted_labels = predict(net, testDataset);
inference_time = toc;
numSamples = size(test_data, 4);
tic;
predicted_labels = classify(net, test_data);
% predicted_labels = predict(net, testDataset);
inference_time = toc;

% Accuracy
accuracy = mean(predicted_labels == test_labels_cat);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Time inference
avg_time = inference_time / numSamples;
fprintf('Total inference time: %.4f seconds\n', inference_time);
fprintf('Average time per image: %.6f seconds\n', avg_time);


