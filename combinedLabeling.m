
function labels = combinedLabeling(semanticSoftmax, edgeSoftmax, threshold)
    % semanticSoftmax: SoftMax output from the semantic segmentation network
    % edgeSoftmax: SoftMax output from the edge detection network
    % threshold: threshold for multiple label assignment

    [M, N, C] = size(semanticSoftmax);
    labels = cell(M, N);

    for i = 1:M
        for j = 1:N
            % Get probabilities from both softmax layers
            semanticProb = squeeze(semanticSoftmax(i, j, :));
            edgeProb = squeeze(edgeSoftmax(i, j, :));

            % Combine the probabilities
            combinedProb = (semanticProb + edgeProb) / 2;
            [maxProb, maxIdx] = max(combinedProb);

            % Assign multiple or single labels
            if maxProb > threshold
                labels{i, j} = maxIdx;
            else
                labels{i, j} = find(combinedProb > threshold);
            end
        end
    end
end
