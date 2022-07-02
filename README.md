% Face_Detection_MATLAB
% Face Detection in MATALB using webcam 


% 1. collecting Data 
% Select folder to save collected data
% create subfolders with names as labels
% rerun the code as many times you wish to collect data on as many faces 
% Download corresponding MATLAB liabrary to allow access to your OS 
% you can scane live or scane pictures on your phone as well 
% with c = 150, we are allowing the system to take 150 pictures 
% more pictures mean more accuracy 

clc
clear all
close all
warning off;
cao=webcam;
faceDetector=vision.CascadeObjectDetector;
c=150;
temp=0;
while true
    e=cao.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
    if(temp>=c)
        break;
    else
    es=imcrop(e,bboxes(1,:));
    es=imresize(es,[227 227]);
    filename=strcat(num2str(temp),'.bmp');
    imwrite(es,filename);
    temp=temp+1;
    imshow(es);
    drawnow;
    end
    else
        imshow(e);
        drawnow;
    end
end

% 2. Training the model
% Select main folder from path with all subfolders of collected data inside it 
% you can name your network whatever you wish, I have named my network myNet1 
% add the number of subfolders in fullyconnectedLayer()
% in place of 'FaceDetection' in line 57, enter your main folders name 

clc
clear all
close all
warning off
g=alexnet;
layers=g.Layers;
layers(23)=fullyConnectedLayer(3);
layers(25)=classificationLayer;
allImages=imageDatastore('FaceDetection','IncludeSubfolders',true, 'LabelSource','foldernames');
opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
myNet1=trainNetwork(allImages,layers,opts);
save myNet1;

% now the system has created a network with the data provided in the main folder 


% 3. Test Model
% Select main folder again 
% Now the system will load the network and try to find a match
% if match is found, it will return subfolders name 
% if no face is detectable it will print 'No Face Detected' 


clc;close;clear
c=webcam;
load myNet1;
faceDetector=vision.CascadeObjectDetector;
while true
    e=c.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
     es=imcrop(e,bboxes(1,:));
    es=imresize(es,[227 227]);
    label=classify(myNet1,es);
    image(e);
    title(char(label));
    drawnow;
    else
        image(e);
        title('No Face Detected');
    end
end

