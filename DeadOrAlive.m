%% Are the Roundworms Alive or Dead?
% (Alive worms are round; dead ones are straight.)

clc;clear;close all

%% Create a datastore to the images.
imds = imageDatastore("WormImages");
%% image labels.
groundtruth = readtable("WormData.csv");
imds.Labels = categorical(groundtruth.Status);
%% View the first few images.

t=tiledlayout(2,2,"TileSpacing","loose")
nexttile
imshow(readimage(imds,1))
nexttile
imshow(readimage(imds,2))
nexttile
imshow(readimage(imds,3))
nexttile

imshow(readimage(imds,4))
exportgraphics(t,'worms.png','Contenttype','image','Resolution',600)


%% Divide data into training (60%) and testing (40%) sets
[trainImgs,testImgs] = splitEachLabel(imds,0.6,"randomized");

%% Create augmented image datastores to preprocess the images.
trainds = augmentedImageDatastore([224 224],trainImgs,"ColorPreprocessing","gray2rgb");
testds = augmentedImageDatastore([224 224],testImgs,"ColorPreprocessing","gray2rgb");

%% Build a network
%% Start with a pretrained network
net = googlenet;
lgraph = layerGraph(net);
%% Take the CNN layer graph and replace the output layers.
newFc = fullyConnectedLayer(2,"Name","new_fc");
lgraph = replaceLayer(lgraph,"loss3-classifier",newFc);
newOut = classificationLayer("Name","new_out");
lgraph = replaceLayer(lgraph,"output",newOut);

%% Set some training options
options = trainingOptions("sgdm","InitialLearnRate", 0.001);

%% Train the network
wormsnet = trainNetwork(trainds,lgraph,options);

%% Evaluate network on test data
%% Make predictions
preds = classify(wormsnet,testds);
Compare with reality
truetest = testImgs.Labels;
nnz(preds == truetest)/numel(preds)
%% View confusion matrix
confusionchart(truetest,preds);
%% View first incorrect classification (if there is one)
idx = find(preds~=truetest);
if ~isempty(idx)
    imshow(readimage(testImgs,idx(1)))
    title(truetest(idx(1)))
end