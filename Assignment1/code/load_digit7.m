function [DigitsTrain,DigitsTest] = load_digit7();

load('digits/digit7.mat');
D = D/255; %% normalize 
D = reshape(permute(reshape(D,[size(D,1),28,28]),[1,3,2]),size(D,1),size(D,2));
DigitsTrain = D(1:2:end,:);
DigitsTest  = D(2:2:end,:);


