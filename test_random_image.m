function test_random_image(filename)
img = imread(filename);
imshow(img);
imagefeatures = double(encode(bag, img));
[bestGuess] = predict(trainedClassifier.ClassificationTree, imagefeatures);
title(sprintf('My best guess is: %s', char(bestGuess)));