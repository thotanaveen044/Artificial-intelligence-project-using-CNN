clc
clear
close all

V= fullfile('C:\Users\naveen\Documents\MATLAB\dog dataset\images\Images\n02085782-Japanese_spaniel');
%V= 'C:\Users\naveen\Documents\MATLAB\New folder\images\Annotation';
%s = fullfile(matlabroot,'toolbox','nnet','nndemos','nndatasets','D'); 


filenames=dir(fullfile(V,'*.jpg'));
count=numel(filenames);
for n=1:count
    f=fullfile(V,filenames(n).name);
    i=imread(f);
    I= imresize(i,[120 120])
    path=strcat('C:\Users\naveen\Documents\MATLAB\demo\',filenames(n).name);
    imwrite(I,path);
end
%s=dir([D,'*.jpg']);
%names={s.name};
%for k= 1:length(s);
    %image=s(k).name;
   % filename=[D 'n' s(k).name];
  %I=resize(filname,[300 300])
    %img1 =imread(filename);
    
    %p=imread(h);
    %I=resize(p,[300 300]);
   % imwrite(I,fullfile('D:','New_folder',H));
%end

imds = imageDatastore(V,'IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(imds);
img = readimage(imds,1);
size(img)



%imds = imageDatastore(D,'IncludeSubfolders',true,'LabelSource','foldernames');
%labelCount = countEachLabel(imds);
%img=readimage(imds,1);
%img = rgb2gray(imresize(readimage(imds,1),[120 120])); 
%j=rgb2gray(imresize(img,[120 120]));
%size(img)

%n= size(img);
%for i:

%grayImage = rgb2gray(j);

%data = rand(20000,10);
% Cross varidation (train: 80%, test: 20%)
%cv = cvpartition(size(data,1),'HoldOut',0.2);
%idx = cv.test;
% Separate to training and test data
%dataTrain = data(~idx,:);
%dataTest  = data(idx,:);




%valid = cvpartition(size(dataTrain,1),'HoldOut',0.1);
%idx1= valid.test;
%dataTrain1= dataTrain(~idx1,:);
%Validation= dataTrain(idx1,:);


figure; 
perm = randperm(185,20); 
for i = 1:20     
    subplot(5,4,i);   
    imshow(imds.Files{perm(i)});
    %imresize(imds,[120 120])
end

numTrainFiles =  140; 
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize'); 
%imdsTrain1= imresize(imdsTrain,[120 120]);
 
 

%[XTrain,YTrain] = dataTrain;

%idx = randperm(size(XTrain,4),4000);
%XValidation = XTrain(:,:,:,idx);
%XTrain(:,:,:,idx) = [];
%YValidation = YTrain(idx);
%YTrain(idx) = [];

layers = [     
    imageInputLayer([120 120 3])          
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer     
    reluLayer         
    maxPooling2dLayer(2,'Stride',2)       
    convolution2dLayer(3,16,'Padding','same')    
    batchNormalizationLayer  
    reluLayer       
    maxPooling2dLayer(2,'Stride',2)    
    convolution2dLayer(3,32,'Padding','same') 
    batchNormalizationLayer  
    reluLayer          
    fullyConnectedLayer(1) 
    softmaxLayer  
    classificationLayer
    ]; 
options = trainingOptions('sgdm','InitialLearnRate',0.01,'MaxEpochs',4,'Shuffle','every-epoch','ValidationData',imdsValidation,'ValidationFrequency',30,'Verbose',false,'Plots','training-progress');

% options = trainingOptions('sgdm','InitialLearnRate',0.01,'MaxEpochs',4,'Shuffle','every-epoch','ValidationData',imdsValidation,'ValidationFrequency',30,'Verbose',false,'Plots','training-progress');

 %options = trainingOptions('sgdm','InitialLearnRate',0.01, 'MaxEpochs',10,'ValidationData',imdsValidation ,'ValidationFrequency',30, 'Verbose',false,'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);
