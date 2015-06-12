clear
imgSets = [imageSet('data/Img4x3/Sample001'),
           imageSet('data/Img4x3/Sample002'),
           imageSet('data/Img4x3/Sample003'),
           imageSet('data/Img4x3/Sample004'),
           imageSet('data/Img4x3/Sample005'),
           imageSet('data/Img4x3/Sample006'),
           imageSet('data/Img4x3/Sample007'),
           imageSet('data/Img4x3/Sample008'),
           imageSet('data/Img4x3/Sample009'),
           imageSet('data/Img4x3/Sample010'),
           imageSet('data/Img4x3/Sample011'),
           imageSet('data/Img4x3/Sample012'),
           imageSet('data/Img4x3/Sample013'),
           imageSet('data/Img4x3/Sample014'),
           imageSet('data/Img4x3/Sample015'),
           imageSet('data/Img4x3/Sample016'),
           imageSet('data/Img4x3/Sample017'),
           imageSet('data/Img4x3/Sample018'),
           imageSet('data/Img4x3/Sample019'),
           imageSet('data/Img4x3/Sample020'),
           imageSet('data/Img4x3/Sample021'),
           imageSet('data/Img4x3/Sample022'),
           imageSet('data/Img4x3/Sample023'),
           imageSet('data/Img4x3/Sample024'),
           imageSet('data/Img4x3/Sample025'),
           imageSet('data/Img4x3/Sample026'),
           imageSet('data/Img4x3/Sample027'),
           imageSet('data/Img4x3/Sample028'),
           imageSet('data/Img4x3/Sample029'),
           imageSet('data/Img4x3/Sample030'),
           imageSet('data/Img4x3/Sample031'),
           imageSet('data/Img4x3/Sample032'),
           imageSet('data/Img4x3/Sample033'),
           imageSet('data/Img4x3/Sample034'),
           imageSet('data/Img4x3/Sample035'),
           imageSet('data/Img4x3/Sample036')];
imgSets(1).Description = '0';
imgSets(2).Description = '1';
imgSets(3).Description = '2';
imgSets(4).Description = '3';
imgSets(5).Description = '4';
imgSets(6).Description = '5';
imgSets(7).Description = '6';
imgSets(8).Description = '7';
imgSets(9).Description = '8';
imgSets(10).Description = '9';
imgSets(11).Description = 'a';
imgSets(12).Description = 'b';
imgSets(13).Description = 'c';
imgSets(14).Description = 'd';
imgSets(15).Description = 'e';
imgSets(16).Description = 'f';
imgSets(17).Description = 'g';
imgSets(18).Description = 'h';
imgSets(19).Description = 'i';
imgSets(20).Description = 'j';
imgSets(21).Description = 'k';
imgSets(22).Description = 'l';
imgSets(23).Description = 'm';
imgSets(24).Description = 'n';
imgSets(25).Description = 'o';
imgSets(26).Description = 'p';
imgSets(27).Description = 'q';
imgSets(28).Description = 'r';
imgSets(29).Description = 's';
imgSets(30).Description = 't';
imgSets(31).Description = 'u';
imgSets(32).Description = 'v';
imgSets(33).Description = 'w';
imgSets(34).Description = 'x';
imgSets(35).Description = 'y';
imgSets(36).Description = 'z';

{ imgSets.Description } % display all labels on one line
[imgSets.Count]         % show the corresponding count of images

minSetCount = min([imgSets.Count]); % determine the smallest amount of images in a category

% Use partition method to trim the set.
imgSets = partition(imgSets, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
[imgSets.Count]

[trainingSets, validationSets] = partition(imgSets, 0.7, 'randomize');

bag = bagOfFeatures(trainingSets);

categoryClassifier = trainImageCategoryClassifier(trainingSets, bag);

confMatrix = evaluate(categoryClassifier, trainingSets);

confMatrix = evaluate(categoryClassifier, validationSets);

% Compute average accuracy
mean(diag(confMatrix));
