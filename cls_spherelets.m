function [label_te, accuracy] = cls_spherelets(X_tr, y_tr, X_te, y_te, k, d)
% spherelets based classification algorithm

%% input: X_tr: training data
%           y_tr: label of training data
%           X_te: testing data
%           y_te: label of testing data
%           k: number of neighbors in k-nn
%           d: intrinsic dimension of the manifold
%% Output: label_te: predicted labels
%                  accuracy: prediction accuracy on testing dataset 


%% History:
%   Didong Li       August 24, 2018, created

%% 
[n_tr,p] = size(X_tr); % p is ambient dimension
[n_te,p] = size(X_te); 
label_te = zeros(n_te,1);
if min(y_tr)==0
    y_tr = y_tr+1;
    y_te = y_te+1;
end
N = max(y_tr); % N is number of groups
accuracy=0; % miss classification rate
wrong = [];
data = cell(N,1);
n_cls = zeros(N,1);
for j =1:N
    data{j} = X_tr(find(y_tr==j),:);
    n_cls(j) = size(data{j},1);
end
n_min = min(n_cls);
j_min = min(find(n_cls==n_min));
if n_min<k
    display(['no enough samples with label=', num2str(j_min)])
    return
end
    
    
for i = 1:n_te
    cls_dist = zeros(N,1); % distance between a testing sample and all groups in training set
    for j = 1:N
         diff = data{j}-ones(n_cls(j),1)*X_te(i,:); 
         dist = sqrt(sum(diff.^2,2)); % distance between the testing sample and each sample in X_j
         [dist, knn_label] = sort(dist);
         knn_label = knn_label(1:k,1);
         knn = data{j}(knn_label,:); % find the k-nearest neighbors within group j
         [c,V,r]=SPCA(knn,d); % do SPCA for these neighbors
         xhat = c+V*V.'*(X_te(i,:).'-c)*r/norm(V*V.'*(X_te(i,:).'-c));
         cls_dist(j) = norm(X_te(i,:)-xhat.'); % distance between the testing point and the spherelet
    end
    [sort_dist,label_te(i)] = min(cls_dist); % assign to the nearest manifold
    if label_te(i)~=y_te(i)
        wrong = [wrong; i];
    else
        accuracy = accuracy + 1;
    end
end

accuracy = accuracy/n_te; 

return

