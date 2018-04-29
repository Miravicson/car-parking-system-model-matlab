%% Car Parking Space Verification Using Image Data
% Image classification involves determining if an image contains some 
% specific object, feature, or activity. The goal of this example is to
% provide a strategy to construct a classifier that can automatically 
% detect if a car parking space has a car or motorcycle parked in it or if
% the space is empty
% This example uses function from the Computer Vision System Toolbox and
% Statistics and Machine Learning

%% Description of the Data
% The dataset contains 3 image categories: Car in car park, Motorcycle in car park, Empty car park
% The images are photos of the different image categoreis that have been taken from different
% angles, positions, and different lighting conditions. These variations make 
% this a challenging task.

% Please note that all data in this example was resized to the same height
% and width dimensions. Montage will not work unless you resize all of the images
% This is done using the custom read function for ImageDatastore :
% readAndResizeImages.m
% You can alter the size of the images by opening that function

%%%


%% Load image data
% This assumes you have a directory: dataset
% with each image categories in a subdirectory
imds = imageDatastore('dataset2',...
    'IncludeSubfolders',true,'LabelSource','foldernames')              %#ok
imds.ReadFcn = @readAndResizeImage;

%% Display Class Names and Counts
tbl = countEachLabel(imds)                                             %#ok
categories = tbl.Label;

%% Display Sampling of Image Data
visImds = splitEachLabel(imds,1,'randomize');

for ii = 1:3 % this assumes 4 categories of scenes
    subplot(3,1,ii);
    imshow(visImds.readimage(ii));
    title(char(visImds.Labels(ii)));    
end

%% Pre-process Training Data: *Feature Extraction using Bag Of Words*
% Bag of features, also known as bag of visual words is one way to extract 
% features from images. To represent an image using this approach, an image 
% can be treated as a document and occurance of visual "words" in images
% are used to generate a histogram that represents an image.
%% Partition 700 images for training and 200 for testing
[training_set, test_set] = crossValidate(imds);

%% Create Visual Vocabulary 
tic
bag = bagOfFeatures(training_set,...
    'VocabularySize',250,'PointSelection','Detector');
picturedata = double(encode(bag, training_set));
toc
%% Visualize Feature Vectors 
img = read(training_set(1), randi(training_set(1).Count));
featureVector = encode(bag, img);

subplot(3,2,1); imshow(img);
subplot(3,2,2); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(training_set(2), randi(training_set(2).Count));
featureVector = encode(bag, img);
subplot(3,2,3); imshow(img);
subplot(3,2,4); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(training_set(3), randi(training_set(3).Count));
featureVector = encode(bag, img);
subplot(3,2,5); imshow(img);
subplot(3,2,6); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

%% Create a Table using the encoded features
CarMotorcycleImgData = array2table(picturedata);
ImageType = categorical(repelem({training_set.Description}', [training_set.Count], 1));
CarMotorcycleImgData.ImageType = ImageType;

%% Use the new features to train a model and assess its performance using 
classificationLearner

%% Test out accuracy on test set!

testSceneData = double(encode(bag, test_set));
testSceneData = array2table(testSceneData,'VariableNames',trainedClassifier.RequiredVariables);
actualSceneType = categorical(repelem({test_set.Description}', [test_set.Count], 1));

predictedOutcome = trainedClassifier.predictFcn(testSceneData);

correctPredictions = (predictedOutcome == actualSceneType);
validationAccuracy = sum(correctPredictions)/length(predictedOutcome) %#ok
return;

%% Visualize how the classifier works
ii = randi(size(test_set,2));
jj = randi(test_set(ii).Count);
img = read(test_set(ii),jj);

imshow(img)
% Add code here to invoke the trained classifier
imagefeatures = double(encode(bag, img));
% Find two closest matches for each feature
[bestGuess, score] = predict(trainedClassifier.ClassificationSVM,imagefeatures);
% Display the string label for img
if strcmp(char(bestGuess),test_set(ii).Description)
	titleColor = [0 0.8 0];
else
	titleColor = 'r';
end
title(sprintf('Best Guess: %s; Actual: %s',...
	char(bestGuess),test_set(ii).Description),...
	'color',titleColor)

