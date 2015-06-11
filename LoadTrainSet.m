function [ trainingSet ] = LoadTrainSet()
    disp('--- Starting the loading of the Trainset ---');
    all_img = importfile('data/all.txt');
    nb_img_per_class = 55;
    crt_class = 0;
    for i=1:size(all_img)
        id = mod(i-1,nb_img_per_class);
        if id == 0
            crt_class = crt_class + 1;
        end
        crt_file = strcat('data/',char(all_img(i))); % create path
        I = imread(crt_file); % load image file
        I = rgb2gray(I); % 3 dimensional to 2 dimensional, required by SURF
        points = detectSURFFeatures(I); % extract surf features
        [features, valid_points] = extractFeatures(I, points);
        trainingSet.class(crt_class).image(id+1).features = features;
    end
    disp('--- Finished the loading of the Trainset ---');
end