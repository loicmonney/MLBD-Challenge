clear
trainingSet = LoadTrainSet();
%load('data/trainingSet_4.mat');

input = []; % the features and all samples
output = []; % which class is the correct answer of a given feature (in input)
%for i=1:size(trainingSet.class,2) % cycle through all classes
for i=1:10 % only image with 0-9
    for j=1:size(trainingSet.class(i).image,2)
        input = [input; trainingSet.class(i).image(j).features];
        %temp = zeros(size(trainingSet.class(i).image(j).features,1),size(trainingSet.class,2)); % matrix nbClass x nbSamples
        temp = zeros(size(trainingSet.class(i).image(j).features,1),10);
        temp = temp';
        temp(i,:) = 1;
        temp = temp';
        output = [output; temp];
    end
end
input = input'; % features are lines, images are columns
output = output';

x = input;
t = output;

% Create a Pattern Recognition Network
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);

% Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

% Train the Network
[net,tr] = train(net,x,t);

model = struct('net',net,'tr',tr);
disp('--- Finished the training of the model for the algorithm ---');

% Test the Network
y = net(x);
e = gsubtract(t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);
performance = perform(net,t,y)

% View the Network
%view(net)

% Plots
% Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
disp('--- Display the confusion matrix ---');
figure, plotconfusion(t,y)
%figure, plotroc(t,y)
%figure, ploterrhist(e)