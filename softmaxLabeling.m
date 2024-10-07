
function labels = softmaxLabeling(softmaxOutput, threshold)
    % softmaxOutput: an MxNxC array where M and N are the image dimensions,
    % and C is the number of classes from the softmax layer.
    % threshold: if the highest probability is above the threshold, assign a single label.
    % Otherwise, assign multiple labels.

    [M, N, C] = size(softmaxOutput);
    labels = cell(M, N);  % Store labels as a cell array

    for i = 1:M
        for j = 1:N
            % Get the probabilities for each class at pixel (i, j)
            prob = squeeze(softmaxOutput(i, j, :));
            [maxProb, maxIdx] = max(prob);

            % If the max probability is above the threshold, assign a single label
            if maxProb > threshold
                labels{i, j} = maxIdx;
            else
                % Assign all labels with probabilities above the threshold
                labels{i, j} = find(prob > threshold);
            end
        end
    end
end
