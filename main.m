

disp('Select a labeling method:');
disp('1: SoftMax Layer-Based Labeling');
disp('2: Semantic Segmentation with Edge Detection');
disp('3: Combination of Semantic Segmentation and Edge Detection');
method = input('Enter method number: ');


softmaxOutput = imread('example 1.png)
semanticEdges = edge(imread('example edge 1.png));  % Example edge detection
edgeSoftmax = rand(100, 100, 5);  % Example random edge softmax output


threshold = 0.3;


switch method
    case 1
        disp('Running SoftMax Layer-Based Labeling...');
        labels = softmaxLabeling(softmaxOutput, threshold);
    case 2
        disp('Running Semantic Segmentation with Edge Detection...');
        labels = semanticEdgeLabeling(softmaxOutput, semanticEdges, threshold);
    case 3
        disp('Running Combination of Semantic Segmentation and Edge Detection...');
        labels = combinedLabeling(softmaxOutput, edgeSoftmax, threshold);
    otherwise
        disp('Invalid method number!');
end

% Display or process the result
disp('Labeling completed. You can now use the "labels" variable.');
