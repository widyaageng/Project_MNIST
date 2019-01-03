clear;clc;

%% %------DATA STRUCTURE--------%

% training image data structure
im_train = (fread(fopen('train-images.idx3-ubyte','r'),inf,'uint8'));
im_train_header = cast(im_train(1:16),'uint8');
im_train_header = reshape(dec2hex(im_train_header).',[8 4])';
im_train_count = hex2dec(im_train_header(2,:));
im_train = reshape(im_train(17:size(im_train)),[28*28 im_train_count]);
clear im_train_header;

% training label data structure
lab_train = (fread(fopen('train-labels.idx1-ubyte','r'),inf,'uint8'));
lab_train = lab_train(9:size(lab_train));
clear lab_train_header;

% test image data structure
im_test = (fread(fopen('t10k-images.idx3-ubyte','r'),inf,'uint8'));
im_test_header = cast(im_test(1:16),'uint8');
im_test_header = reshape(dec2hex(im_test_header).',[8 4])';
im_test_count = hex2dec(im_test_header(2,:));
im_test = reshape(im_test(17:size(im_test)), [28*28 im_test_count]);
clear im_test_header

% test label data structure
lab_test = (fread(fopen('t10k-labels.idx1-ubyte','r'),inf,'uint8'));
lab_test = lab_test(9:size(lab_test));
clear lab_test_header;

%% ----------------STATISTICS---------------

% sorting the train image into ascending label, image vector follows
% accordingly
temp_lab_train = lab_train;
[temp_lab_train,train_idx] = sort(lab_train);
% dump_train = im_train(:,train_idx(1));
% for i=2:im_train_count
%     dump_train = [dump_train im_train(:,train_idx(i))];
% end
load('dump_train');

% Finding class mean vector
last_Cdat = [0 find(temp_lab_train==0,1,'last')];
im_train_Cmean = [mean(dump_train(:,last_Cdat(1)+1:(last_Cdat(2))),2)];
for i=2:10
    last_Cdat = [last_Cdat find(temp_lab_train==(i-1),1,'last')];
    im_train_Cmean = [im_train_Cmean mean(dump_train(:,1+last_Cdat(i):last_Cdat(i+1)),2)];
end
    
% Finding training image mean,  total mean vector   
im_train_mean = mean(im_train,2);

% Finding prior class probability
im_class_pr = zeros(1,size(im_train_Cmean,2));
for j=1:size(im_train_Cmean,2)
    im_class_pr(j) = (last_Cdat(j+1) - last_Cdat(j))/im_train_count;
end
    
% Calculating within class scatter, Sw
train_Sw = zeros(size(im_train,1),size(im_train,1));
im_class_pr = im_class_pr*0 + 1;
for i=2:11
    im_vec_diff_mat = 0*im_train_Cmean(:,i-1)*im_train_Cmean(:,i-1)';
    for j=(last_Cdat(i-1)+1):last_Cdat(i)
        im_vec_diff_mat = im_vec_diff_mat + (dump_train(:,j) - im_train_Cmean(:,i-1))*(dump_train(:,j) - im_train_Cmean(:,i-1))';
    end
    train_Sw = train_Sw + im_vec_diff_mat;
end

% Calculating between class scatter, Sb
im_meanvec_diff = im_train_Cmean(:,1) - im_train_mean;
train_Sb = im_meanvec_diff*im_meanvec_diff';
for i=2:10
    im_meanvec_diff = im_train_Cmean(:,i) - im_train_mean;
    train_Sb = train_Sb + im_meanvec_diff*im_meanvec_diff';
end

% Calculating J function (max), biggest eigenvalue that corresponds to
% optimal projection line, biggest ratio between a big between-class
% scatter and small within-class scatter
[LDA_eigv,LDA_eig] = eig(train_Sw\train_Sb);
[LDA_sort_eig,LDA_idx] = sort(diag(LDA_eig),'descend');

%% LDA reduction using 2 optimal projection vector W

% optimal projection line
W_opt2 = LDA_eigv(:,LDA_idx(1:2));

% ration maximized
J2 = det(W_opt2'*train_Sb*W_opt2)/det(W_opt2'*train_Sw*W_opt2);

% training grouped image projection
im_train_proj2 = W_opt2'*dump_train;

% Initiate distinct color for scatter plot
RGB = [0.3686 0.3098    0.6353;
    0.2005    0.5593    0.7380;
    0.4558    0.7897    0.6458;
    0.7616    0.9058    0.6299;
    0.9277    0.9583    0.6442;
    0.9820    0.9206    0.6372;
    0.9942    0.7583    0.4312;
    0.9684    0.4799    0.2723;
    0.8525    0.2654    0.3082;
    0.6196    0.0039    0.2588];

% Finding the index of last column data on each class and scatter plot it
figure;hold;
for i=1:10
    scatter(im_train_proj2(1,last_Cdat(i)+1:last_Cdat(i+1)),im_train_proj2(2,last_Cdat(i)+1:last_Cdat(i+1)),2,RGB(i,:));
end
colormap(RGB);
col_handle = colorbar('Location', 'EastOutside', 'YTick',[1/20 1/20+1/10 1/20+2/10 1/20+3/10 1/20+4/10 1/20+5/10 1/20+6/10 1/20+7/10 1/20+8/10 1/20+9/10]);
col_handle.TickLabels = {'0','1','2','3','4','5','6','7','8','9'};
grid;
xlabel('1st LDA projector');
ylabel('2nd LDA projector');
hold off;

%% LDA reduction using 3 optimal projection vector W

% optimal projection line
W_opt3 = LDA_eigv(:,LDA_idx(1:3));

% ration maximized
J3 = det(W_opt3'*train_Sb*W_opt3)/det(W_opt3'*train_Sw*W_opt3);

% training grouped image projection
im_train_proj3 = W_opt3'*dump_train;

% Initiate distinct color for scatter plot
RGB = [0.3686 0.3098    0.6353;
    0.2005    0.5593    0.7380;
    0.4558    0.7897    0.6458;
    0.7616    0.9058    0.6299;
    0.9277    0.9583    0.6442;
    0.9820    0.9206    0.6372;
    0.9942    0.7583    0.4312;
    0.9684    0.4799    0.2723;
    0.8525    0.2654    0.3082;
    0.6196    0.0039    0.2588];

% Finding the index of last column data on each class and scatter plot it
figure;hold;
for i=1:10
    scatter3(im_train_proj3(1,last_Cdat(i)+1:last_Cdat(i+1)),im_train_proj3(2,last_Cdat(i)+1:last_Cdat(i+1)),im_train_proj3(3,last_Cdat(i)+1:last_Cdat(i+1)),2,RGB(i,:));
end
colormap(RGB);
col_handle = colorbar('Location', 'EastOutside', 'YTick',[1/20 1/20+1/10 1/20+2/10 1/20+3/10 1/20+4/10 1/20+5/10 1/20+6/10 1/20+7/10 1/20+8/10 1/20+9/10]);
col_handle.TickLabels = {'0','1','2','3','4','5','6','7','8','9'};
grid;
xlabel('1st LDA projector');
ylabel('2nd LDA projector');
zlabel('3rd LDA projector');
view(3);
hold off;

%% LDA reduction using 40 optimal projection vector W
KNN = [3 5 8 10 15 20 25 30 40 50];
knn_label_40 = zeros(10,1)*lab_test';
% optimal projection line
W_opt40 = LDA_eigv(:,LDA_idx(1:2));

% ration maximized
J40 = det(W_opt40'*train_Sb*W_opt40)/det(W_opt40'*train_Sw*W_opt40);

% training image projection
im_train_proj40 = W_opt40'*im_train;

% test image projection
im_test_proj40 = W_opt40'*im_test;
% Classifying 40PCs projected test data to existing pool of projected train
% data
for k=1:length(KNN)
    h = waitbar(0,'Please wait...');
    index = 0;
    % knn_label_40 = 0*lab_test;
    for i=1:size(im_test_proj40,2)
        dist40 = zeros(1,length(im_train_proj40));
        for j=1:size(im_train_proj40,2)
            dist40(j) = sqrt((im_test_proj40(:,i)-im_train_proj40(:,j))'*(im_test_proj40(:,i)-im_train_proj40(:,j)));
            index = index + 1;
        end
        [dist40, dist_idx40] = sort(dist40);
        dist_idx40 = dist_idx40(1:KNN);
        knn_label_40(k,i) = mode(lab_train(dist_idx40));
        waitbar(index/(length(im_test_proj40)*length(im_train_proj40)),h,sprintf('Classifying test image KNN...%2.1f%%',100*index/(length(im_test_proj40)*length(im_train_proj40))));
    end
    close(h);
    KNN(k);
end