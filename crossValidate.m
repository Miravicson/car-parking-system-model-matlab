function [tr_set,test_set] = crossValidate(dsObj)
% as of 16a, bagOfFeatures still requires an imageSet object to run. This
% is on the roadmap to change in the future, but for now, we need to
% convert this to an imageSet object! 

image_location = fileparts(dsObj.Files{1});

imset = imageSet(strcat(image_location,'\..'),'recursive');
[tr_set,test_set] = imset.partition([0.8,0.2]); %splits the imageset into 80% for training and 20% for testing
end