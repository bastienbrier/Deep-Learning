[Dtrain,Dtest]  = load_digit7;

whos
[nsamples,ndimensions] = size(Dtrain);

meanDigit =0;
for k=1:nsamples
    meanDigit = meanDigit + Dtrain(k,:)/nsamples;
end
%% simpler & faster
meanDigit = mean(Dtrain,1)';
meanImage = reshape(meanDigit,[28,28]);
figure,imshow(meanImage);

covDigits = 0;
for k=1:nsamples
    covDigits = covDigits + ((Dtrain(k,:)-meanDigit')'*(Dtrain(k,:)-meanDigit'))/(nsamples-1);
end
covDigitsMatlab = cov(Dtrain);
%% make sure covDigitsMatlab = your covDigits
figure,
subplot(1,2,1); imagesc(covDigitsMatlab)
subplot(1,2,2); imagesc(covDigits)

%% get top-5 eigenvectors
[eigvec,eigvals] = eigs(covDigitsMatlab,5);

figure,
subplot(1,3,1); imshow(reshape(eigvec(:,1),[28,28]),[])
subplot(1,3,2); imshow(reshape(eigvec(:,2),[28,28]),[])
subplot(1,3,3); imshow(reshape(eigvec(:,2),[28,28]),[])

for basis_idx = [1:3]
    factors =[-2,0,2];
    figure,
    for k=1:3
        imshow(reshape(meanDigit + 2*factors(k)*eigvec(:,basis_idx),[28,28]))
        pause
    end
end

