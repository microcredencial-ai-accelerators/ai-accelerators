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

%% Define Neural Network models
numClasses = length(unique(train_labels));
model_type = 'fc' % 'fc' or 'conv'

if model_type == 'fc'

    layers = [
        imageInputLayer([28 28 1]) % Tamaño de entrada
        flattenLayer
        fullyConnectedLayer(128)
        reluLayer
        fullyConnectedLayer(32)
        reluLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];

elseif model_type == 'conv'

    layers = [
        imageInputLayer([28 28 1]) % Tamaño de entrada
        convolution2dLayer(3, 32, 'Padding', 'same')
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        convolution2dLayer(3, 64, 'Padding', 'same')
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)
        
        flattenLayer
        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(numClasses)
        softmaxLayer
        classificationLayer
    ];

else
    layers = []
end

%% Train

options = trainingOptions('adam', ...
    'MaxEpochs', 5, ...
    'MiniBatchSize', 64, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(trainDataset, layers, options);

info = analyzeNetwork(net);

%% Evaluate model

% Test dataset
test_data = reshape(test_data, 28, 28, 1, []);
test_data = single(test_data);
test_labels_cat = categorical(test_labels);
testDataset = augmentedImageDatastore([28 28 1], test_data);

% Predict

numSamples = size(test_data, 4);
tic;
predicted_labels = classify(net, testDataset);
inference_time = toc;

% Accuracy
accuracy = mean(predicted_labels == test_labels_cat);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% Time inference
avg_time = inference_time / numSamples;
fprintf('Total inference time: %.4f seconds\n', inference_time);
fprintf('Average time per image: %.6f seconds\n', avg_time);


