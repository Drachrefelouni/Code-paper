
function labels = semanticEdgeLabeling(softmaxOutput, semanticEdges, threshold)
    % softmaxOutput: the same as in Method 1
    % semanticEdges: a binary edge map indicating the edges between semantic classes
    % threshold: a threshold for multiple label assignment

    [M, N, C] = size(softmaxOutput);
    labels = cell(M, N);

    for i = 1:M
        for j = 1:N
            % Get the probabilities for each class at pixel (i, j)
            prob = squeeze(softmaxOutput(i, j, :));
            [maxProb, maxIdx] = max(prob);

            if semanticEdges(i, j)
                % If it's an edge pixel, assign multiple labels
                labels{i, j} = find(prob > threshold);
            else
                % Non-edge pixels get a single label
                labels{i, j} = maxIdx;
            end
        end
    end
end

 
