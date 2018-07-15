clc;
close all;
clear;
%% 一、图像预处理
% imgori = rgb2gray(im2double(imread('picture.jpg')));
imgori =imread('大作业测试图片1.jpg');
figure(1);imshow(imgori);title('原图');
%同态滤波
img_gray=rgb2gray(imgori);
I=im2double(img_gray); 
[M,N]=size(I);
P = 2*M;
Q = 2*N; 
I=log(I+1);
FI=fft2(I,P,Q);    %傅里叶变换，并扩大尺寸填充0
% 滤波器
rL=0.9;    %0.7
rH=1.0;   %0.9
c=10;       %锐化参数5
D0=50;  %100
[v, u] = meshgrid(1:Q, 1:P);
u = u - floor(P/2); 
v = v - floor(Q/2); 
D = u.^2 + v.^2;  % 距离的平方
H = 1-exp(-c*(D./D0^2));    %高斯滤波器
H = (rH - rL)*H + rL;    
H=ifftshift(H);  %对H做反中心化
I3=real(ifft2(H.*FI));  %傅里叶逆变换
I4=I3(1:M, 1:N);  %截取一部分
img_tong=exp(I4)-1;  %取指数
figure(2);
imshow(img_tong);
img = medfilt2(img_tong);
figure(3);
imshow(img);
%% 提取文本去除噪声
level = graythresh(img);    
A = imcomplement(im2bw(img,level));  %大津阈值分割后图像取反
A = medfilt2(A,[3 3]);  %中值滤波
%figure;imshow(A);
[height,width] = size(A);
%初步去除图片上非文字部分（如阴影等）
th = height*width/1000;    %去除大阴影的阈值
CC = bwconncomp(A,8);
area = cellfun(@numel,CC.PixelIdxList);  %像素统计（每个连通区域的像素个数）
idx = (area>=th);        %去除非文字部分
idx0 = find(idx == 1);         %得到第几个目标是大目标
[h,w] = size(idx0);
for i = 1:w
    A(CC.PixelIdxList{idx0(i)}) = 0;    %每个细小目标值赋为0
end
% yh=height*width/100000;
% idx_0=(area<yh);
% idx_00=find(idx_0==1);
% [a,b]=size(idx_00);
% for j=1:b
%     A(CC.PixelIdxList{idx_00(j)}) = 0; 
% end
figure(4);imshow(A)


% MN=[4,40];
% se=strel('rectangle',MN);
% B=imdilate(A,se);%图像A1被结构元素B膨胀，背景置0，文字置1
se = strel('disk',45); %膨胀运算
B = imdilate(A,se); 
figure(5);imshow(B);
%进一步去除非文字部分的小目标
CC2 = bwconncomp(B);
numPixels2 = cellfun(@numel,CC2.PixelIdxList);  %像素统计（每个连通区域的像素个数）
max2 = max(numPixels2);        %得到文字区域（像素最多的连通域）
position = find(numPixels2 == max2); 
renew_img = zeros(height,width);
renew_img(CC2.PixelIdxList{position}) = A(CC2.PixelIdxList{position});    %将文字区域的图像提取出来
figure(6);imshow(renew_img);
% CC5 = bwconncomp(renew_img);
% numPixels = cellfun(@numel,CC5.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% max3 = max(numPixels);
% position = find(numPixels == max3); 
% renew_img(CC5.PixelIdxList{position}) = 0;
% figure(7);imshow(renew_img);
%% 标记每行文字
se = strel('rectangle',[5 45]); %膨胀运算
C = imdilate(renew_img,se); 
L = bwconncomp(C);   %更新连通域
%   stats = regionprops('table',bw,'Centroid', ...
%                              'MajorAxisLength','MinorAxisLength');
%   
%           % Get centers and radii of the circles
%           centers = stats.Centroid;
%           diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
%           radii = diameters/2;
%   
%           % Plot the circles
%           hold on
%           viscircles(centers,radii);
%           hold off
stats = regionprops(L,'centroid');   %STATS = regionprops(L,properties)该函数用来测量标注矩阵L中每一个标注区域的一系列属性。获得中心点
centroids = cat(1,stats.Centroid);   %cat连接数组
figure(7);imshow(C);hold on;plot(centroids(:,1),centroids(:,2),'g*');hold off
[L,num] = bwlabel(C);  %num代表有几行文字
%% 计算偏移量
D = C;
for i=1:2:round(1/5*width)   %区域分割细度
    D(:,i) = 0;
end
for i=round(1/5*width):10:1/3*width
    D(:,i)=0;
end
for i=1/3*width:50:width
    D(:,i)=0;
end
CC_new = bwconncomp(D);
stats_new = regionprops(CC_new,'centroid');   %获得中心点
centroid_new = cat(1,stats_new.Centroid);
[hc,wc] = size(centroid_new);  %hc代表有多少个中心点
centr_inf = zeros(hc,wc+2);    %wc=2,第三列是位于第几行的信息
centr_inf(1:hc,1:2) = centroid_new;
for i = 1:hc
    for j = 1:num
        if (L(round(centroid_new(i,2)),round(centroid_new(i,1))) == j)   %L是每行信息
            centr_inf(i,3) = j;
           % fprintf('i=%d,j=%d\n',i,j);
        end
    end
end
%% 存储每个中心点的偏移量
for i = 1:num
    pos = find(centr_inf(:,3) == i);
    centr_inf(pos,4) = centroids(i,2)-centr_inf(pos,2);    %存储偏移量
    a = centr_inf(pos,4);
%    id1 = find(a>=mean(a));     %分段平滑
%    id2 = find(a<mean(a));
%    a(id1) = smoothts(a(id1),'e',100);
%    a(id2) = smoothts(a(id2),'e',100);
%    a(id2) = medfilt1(a(id2),50);

    a = smoothts(a,'e',50);  %单独平滑
    a = medfilt1(a,10);
    centr_inf(pos,4) = a;
    fprintf('%d\n',i);
end
figure(8);imshow(D);hold on;plot(centr_inf(:,1),centr_inf(:,2),'g*');hold off
%% 非线性畸变校正
finalimg = zeros(height,width);
[L_D,num_D] = bwlabel(D);
STATS2 = regionprops(L_D,'PixelList');%PixelList 存储的坐标是列在前，行在后
mask = zeros(height,width);
for i = 1:num_D
    [h,w] = size(STATS2(i).PixelList);
    %maxf = max(STATS2(i).PixelList(:,1));
    %len = length(find(STATS2(i).PixelList(:,1) == maxf));
    statsnow = [STATS2(i).PixelList;[STATS2(i).PixelList(1:h,1)+1,STATS2(i).PixelList((1:h),2)]];
   %[h,w] = size(statsnow);
    %for j = 1:h
        finalimg(statsnow(:,2)+round(centr_inf(i,4)),statsnow(:,1)) = renew_img(statsnow(:,2),statsnow(:,1));%行，列
   % end
end
figure(9);imshow(imcomplement(finalimg));
imwrite(imcomplement(finalimg),'myphoto5.jpg')




% clc;
% close all;
% clear;
% %% 一、图像预处理
% imgori =imread('picture2.jpg');
% figure(1);imshow(imgori);title('原图');
% %同态滤波
% img_gray=rgb2gray(imgori);
% I=im2double(img_gray); 
% [M,N]=size(I);
% P = 2*M;
% Q = 2*N; 
% I=log(I+1);
% FI=fft2(I,P,Q);    %傅里叶变换，并扩大尺寸填充0
% % 滤波器
% rL=0.15;    %0.7
% rH=0.25;   %0.9
% c=10;       %锐化参数5
% D0=50;  %100
% [v, u] = meshgrid(1:Q, 1:P);
% u = u - floor(P/2); 
% v = v - floor(Q/2); 
% D = u.^2 + v.^2;  % 距离的平方
% H = 1-exp(-c*(D./D0^2));    %高斯滤波器
% H = (rH - rL)*H + rL;    
% H=ifftshift(H);  %对H做反中心化
% I3=real(ifft2(H.*FI));  %傅里叶逆变换
% I4=I3(1:M, 1:N);  %截取一部分
% img_tong=exp(I4)-1;  %取指数
% figure(2);
% imshow(img_tong);
% img = medfilt2(img_tong);
% figure(3);
% imshow(img);
% %% 提取文本去除噪声
% level = graythresh(img);    
% A = imcomplement(im2bw(img,level));  %大津阈值分割后图像取反
% A = medfilt2(A,[3 3]);  %中值滤波
% %figure;imshow(A);
% [height,width] = size(A);
% %初步去除图片上非文字部分（如阴影等）
% th = height*width/1000;    %去除大阴影的阈值
% CC = bwconncomp(A,8);
% area = cellfun(@numel,CC.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% idx = (area>=th);        %去除非文字部分
% idx0 = find(idx == 1);         %得到第几个目标是大目标
% [h,w] = size(idx0);
% for i = 1:w
%     A(CC.PixelIdxList{idx0(i)}) = 0;    %每个细小目标值赋为0
% end
% % yh=height*width/100000;
% % idx_0=(area<yh);
% % idx_00=find(idx_0==1);
% % [a,b]=size(idx_00);
% % for j=1:b
% %     A(CC.PixelIdxList{idx_00(j)}) = 0; 
% % end
% figure(4);imshow(A)
% 
% 
% % MN=[4,40];
% % se=strel('rectangle',MN);
% % B=imdilate(A,se);%图像A1被结构元素B膨胀，背景置0，文字置1
% se = strel('disk',40); %膨胀运算
% B = imdilate(A,se); 
% figure(5);imshow(B);
% %进一步去除非文字部分的小目标
% CC2 = bwconncomp(B);
% numPixels2 = cellfun(@numel,CC2.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% max2 = max(numPixels2);        %得到文字区域（像素最多的连通域）
% position = find(numPixels2 == max2); 
% renew_img = zeros(height,width);
% renew_img(CC2.PixelIdxList{position}) = A(CC2.PixelIdxList{position});    %将文字区域的图像提取出来
% figure(6);imshow(renew_img);
% % CC5 = bwconncomp(renew_img);
% % numPixels = cellfun(@numel,CC5.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% % max3 = max(numPixels);
% % position = find(numPixels == max3); 
% % renew_img(CC5.PixelIdxList{position}) = 0;
% % figure(7);imshow(renew_img);
% %% 标记每行文字
% se = strel('rectangle',[5 40]); %膨胀运算
% C = imdilate(renew_img,se); 
% L = bwconncomp(C);   %更新连通域
% %   stats = regionprops('table',bw,'Centroid', ...
% %                              'MajorAxisLength','MinorAxisLength');
% %   
% %           % Get centers and radii of the circles
% %           centers = stats.Centroid;
% %           diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
% %           radii = diameters/2;
% %   
% %           % Plot the circles
% %           hold on
% %           viscircles(centers,radii);
% %           hold off
% stats = regionprops(L,'centroid');   %STATS = regionprops(L,properties)该函数用来测量标注矩阵L中每一个标注区域的一系列属性。获得中心点
% centroids = cat(1,stats.Centroid);   %cat连接数组
% figure(7);imshow(C);hold on;plot(centroids(:,1),centroids(:,2),'g*');hold off
% [L,num] = bwlabel(C);  %num代表有几行文字
% %% 计算偏移量
% D = C;
% for i=1:10:width   %区域分割细度
%     D(:,i) = 0;
% end
% 
% CC_new = bwconncomp(D);
% stats_new = regionprops(CC_new,'centroid');   %获得中心点
% centroid_new = cat(1,stats_new.Centroid);
% [hc,wc] = size(centroid_new);  %hc代表有多少个中心点
% centr_inf = zeros(hc,wc+2);    %wc=2,第三列是位于第几行的信息
% centr_inf(1:hc,1:2) = centroid_new;
% for i = 1:hc
%     for j = 1:num
%         if (L(round(centroid_new(i,2)),round(centroid_new(i,1))) == j)   %L是每行信息
%             centr_inf(i,3) = j;
%            % fprintf('i=%d,j=%d\n',i,j);
%         end
%     end
% end
% %% 存储每个中心点的偏移量
% for i = 1:num
%     pos = find(centr_inf(:,3) == i);
%     centr_inf(pos,4) = centroids(i,2)-centr_inf(pos,2);    %存储偏移量
%     a = centr_inf(pos,4);
% %    id1 = find(a>=mean(a));     %分段平滑
% %    id2 = find(a<mean(a));
% %    a(id1) = smoothts(a(id1),'e',100);
% %    a(id2) = smoothts(a(id2),'e',100);
% %    a(id2) = medfilt1(a(id2),50);
% 
%     a = smoothts(a,'e',50);  %单独平滑
%     a = medfilt1(a,10);
%     centr_inf(pos,4) = a;
%     %fprintf('%d\n',i);
% end
% figure(8);imshow(D);hold on;plot(centr_inf(:,1),centr_inf(:,2),'g*');hold off
% %% 非线性畸变校正
% finalimg = zeros(height,width);
% [L_D,num_D] = bwlabel(D);
% STATS2 = regionprops(L_D,'PixelList');%PixelList 存储的坐标是列在前，行在后
% mask = zeros(height,width);
% for i = 1:num_D
%     [h,w] = size(STATS2(i).PixelList);
%     %maxf = max(STATS2(i).PixelList(:,1));
%     %len = length(find(STATS2(i).PixelList(:,1) == maxf));
%     statsnow = [STATS2(i).PixelList;[STATS2(i).PixelList(1:h,1)+1,STATS2(i).PixelList((1:h),2)]];
%    %[h,w] = size(statsnow);
%     %for j = 1:h
%         finalimg(statsnow(:,2)+round(centr_inf(i,4)),statsnow(:,1)) = renew_img(statsnow(:,2),statsnow(:,1));%行，列
%    % end
% end
% figure(9);imshow(imcomplement(finalimg));
% imwrite(imcomplement(finalimg),'myphoto4.jpg')
% 
















% clc;
% close all;
% clear;
% %% 一、图像预处理
% imgori = rgb2gray(im2double(imread('2.jpg')));
% figure(1);imshow(imgori);title('原图');
% % %同态滤波
% % img_gray=rgb2gray(imgori);
% % I=im2double(img_gray); 
% % [M,N]=size(I);
% % P = 2*M;
% % Q = 2*N; 
% % I=log(I+1);
% % FI=fft2(I,P,Q);    %傅里叶变换，并扩大尺寸填充0
% % % 滤波器
% % rL=0.15;    %0.7
% % rH=0.25;   %0.9
% % c=10;       %锐化参数5
% % D0=50;  %100
% % [v, u] = meshgrid(1:Q, 1:P);
% % u = u - floor(P/2); 
% % v = v - floor(Q/2); 
% % D = u.^2 + v.^2;  % 距离的平方
% % H = 1-exp(-c*(D./D0^2));    %高斯滤波器
% % H = (rH - rL)*H + rL;    
% % H=ifftshift(H);  %对H做反中心化
% % I3=real(ifft2(H.*FI));  %傅里叶逆变换
% % I4=I3(1:M, 1:N);  %截取一部分
% % img_tong=exp(I4)-1;  %取指数
% % figure(2);
% %imshow(imgori);
% img = medfilt2(imgori);
% figure(3);
% imshow(img);
% %% 提取文本去除噪声
% level = graythresh(img);    
% A = imcomplement(im2bw(img,level));  %大津阈值分割后图像取反
% A = medfilt2(A,[3 3]);  %中值滤波
% %figure;imshow(A);
% [height,width] = size(A);
% %初步去除图片上非文字部分（如阴影等）
% th = height*width/1000;    %去除大阴影的阈值
% CC = bwconncomp(A,8);
% area = cellfun(@numel,CC.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% idx = (area>=th);        %去除非文字部分
% idx0 = find(idx == 1);         %得到第几个目标是大目标
% [h,w] = size(idx0);
% for i = 1:w
%     A(CC.PixelIdxList{idx0(i)}) = 0;    %每个细小目标值赋为0
% end
% % yh=height*width/100000;
% % idx_0=(area<yh);
% % idx_00=find(idx_0==1);
% % [a,b]=size(idx_00);
% % for j=1:b
% %     A(CC.PixelIdxList{idx_00(j)}) = 0; 
% % end
% figure(4);imshow(A)
% 
% 
% % MN=[4,40];
% % se=strel('rectangle',MN);
% % B=imdilate(A,se);%图像A1被结构元素B膨胀，背景置0，文字置1
% se = strel('disk',50); %膨胀运算
% B = imdilate(A,se); 
% figure(5);imshow(B);
% %进一步去除非文字部分的小目标
% CC2 = bwconncomp(B);
% numPixels2 = cellfun(@numel,CC2.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% max2 = max(numPixels2);        %得到文字区域（像素最多的连通域）
% position = find(numPixels2 == max2); 
% renew_img = zeros(height,width);
% renew_img(CC2.PixelIdxList{position}) = A(CC2.PixelIdxList{position});    %将文字区域的图像提取出来
% figure(6);imshow(renew_img);
% % CC5 = bwconncomp(renew_img);
% % numPixels = cellfun(@numel,CC5.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% % max3 = max(numPixels);
% % position = find(numPixels == max3); 
% % renew_img(CC5.PixelIdxList{position}) = 0;
% % figure(7);imshow(renew_img);
% %% 标记每行文字
% se = strel('rectangle',[5 40]); %膨胀运算
% C = imdilate(renew_img,se); 
% L = bwconncomp(C);   %更新连通域
% %   stats = regionprops('table',bw,'Centroid', ...
% %                              'MajorAxisLength','MinorAxisLength');
% %   
% %           % Get centers and radii of the circles
% %           centers = stats.Centroid;
% %           diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
% %           radii = diameters/2;
% %   
% %           % Plot the circles
% %           hold on
% %           viscircles(centers,radii);
% %           hold off
% stats = regionprops(L,'centroid');   %STATS = regionprops(L,properties)该函数用来测量标注矩阵L中每一个标注区域的一系列属性。获得中心点
% centroids = cat(1,stats.Centroid);   %cat连接数组
% figure(7);imshow(C);hold on;plot(centroids(:,1),centroids(:,2),'g*');hold off
% [L,num] = bwlabel(C);  %num代表有几行文字
% %% 计算偏移量
% D = C;
% for i=1:10:width   %区域分割细度
%     D(:,i) = 0;
% end
% 
% CC_new = bwconncomp(D);
% stats_new = regionprops(CC_new,'centroid');   %获得中心点
% centroid_new = cat(1,stats_new.Centroid);
% [hc,wc] = size(centroid_new);  %hc代表有多少个中心点
% centr_inf = zeros(hc,wc+2);    %wc=2,第三列是位于第几行的信息
% centr_inf(1:hc,1:2) = centroid_new;
% for i = 1:hc
%     for j = 1:num
%         if (L(round(centroid_new(i,2)),round(centroid_new(i,1))) == j)   %L是每行信息
%             centr_inf(i,3) = j;
%            % fprintf('i=%d,j=%d\n',i,j);
%         end
%     end
% end
% %% 存储每个中心点的偏移量
% for i = 1:num
%     pos = find(centr_inf(:,3) == i);
%     centr_inf(pos,4) = centroids(i,2)-centr_inf(pos,2);    %存储偏移量
%     a = centr_inf(pos,4);
% %    id1 = find(a>=mean(a));     %分段平滑
% %    id2 = find(a<mean(a));
% %    a(id1) = smoothts(a(id1),'e',100);
% %    a(id2) = smoothts(a(id2),'e',100);
% %    a(id2) = medfilt1(a(id2),50);
% 
%     a = smoothts(a,'e',50);  %单独平滑
%     a = medfilt1(a,10);
%     centr_inf(pos,4) = a;
%     %fprintf('%d\n',i);
% end
% figure(8);imshow(D);hold on;plot(centr_inf(:,1),centr_inf(:,2),'g*');hold off
% %% 非线性畸变校正
% finalimg = zeros(height,width);
% [L_D,num_D] = bwlabel(D);
% STATS2 = regionprops(L_D,'PixelList');%PixelList 存储的坐标是列在前，行在后
% mask = zeros(height,width);
% for i = 1:num_D
%     [h,w] = size(STATS2(i).PixelList);
%     %maxf = max(STATS2(i).PixelList(:,1));
%     %len = length(find(STATS2(i).PixelList(:,1) == maxf));
%     statsnow = [STATS2(i).PixelList;[STATS2(i).PixelList(1:h,1)+1,STATS2(i).PixelList((1:h),2)]];
%    %[h,w] = size(statsnow);
%     %for j = 1:h
%         finalimg(statsnow(:,2)+round(centr_inf(i,4)),statsnow(:,1)) = renew_img(statsnow(:,2),statsnow(:,1));%行，列
%    % end
% end
% figure(9);imshow(imcomplement(finalimg));
% imwrite(imcomplement(finalimg),'myphoto4.jpg')











% % % clc;
% % % clear all;
% % % close all;
% % % imgSrc = rgb2gray(im2double(imread('picture.jpg')));
% % % img = medfilt2(imgSrc); 
% % % %imwrite(imgGray,'E:\01灰度图.jpg');
% % % %figure(1);subplot(121);imshow(imgSrc);title('原图');
% % % %figure(1);subplot(122);imshow(imgGray);title('灰度图');
% % % level = graythresh(img);    
% % % A = imcomplement(im2bw(img,level));  %大津阈值分割后图像取反
% % % %figure;imshow(A);
% % % % thresh=100/255;
% % % % imgbw=im2bw(imgGray,thresh);
% % % % imgbw=medfilt2(imgbw,[7 7]);
% % % %imwrite(imgbw,'E:\02二值图.jpg');
% % % figure(1);subplot(111);imshow(A);title('二值图');
% % % imgEdge=edge(A,'canny');   
% % % figure(2);imshow(imgEdge);
% % % imgsmalledeg=bwmorph(imgEdge,'thin',Inf);
% % % figure(3);imshow(imgsmalledeg);
% % % 
% % % [row col]=size(imgsmalledeg);
% % % imgconfine=zeros(row,col);
% % % ytop(col)=0;
% % % ybot(col)=0;
% % % for j=20:col-20
% % %     for i=20:row/4
% % %         if imgsmalledeg(i,j)==1
% % %             imgconfine(i,j)=255;
% % %             ytop(j)=i;
% % %             break;
% % %         end
% % %     end
% % %     for i=row-20:-1:row/4*3
% % %         if imgsmalledeg(i,j)==1
% % %             imgconfine(i,j)=255;
% % %             ybot(j)=i;
% % %             break;
% % %         end
% % %     end
% % % end
% % % figure(4);imshow(imgconfine);
% % % 
% % % top_dvalue(col-500)=0;
% % % top_mvalue(col-500)=0;
% % % top_ratevalue(col-500)=0;
% % % bot_ratevalue(col-500)=0;
% % % top_5poing(col-500)=0;
% % % for i=col-25:-1:col-500
% % %   top_mvalue(i)=(ytop(i)+ytop(i-1)+ytop(i-2)+ytop(i+1)+ytop(i+2))/5;
% % %   top_ratevalue(i)=abs((ytop(i+5)-ytop(i))-(ytop(i)-ytop(i-5)));
% % %   bot_ratevalue(i)=abs((ybot(i+5)-ybot(i))-(ybot(i)-ybot(i-5)));
% % %   top_5poing(i)=ytop(i)-ytop(i-5);
% % % end
% % % for i=col-25:-1:col-500
% % %  top_dvalue(i)=ytop(i)-top_mvalue(i);
% % % end
% % % figure(7);subplot(221),bar(top_mvalue);
% % % figure(7);subplot(222),bar(top_5poing);
% % % figure(7);subplot(223),bar(top_ratevalue);
% % % figure(7);subplot(224),bar(top_dvalue);
% % % 
% % % 
% % % Maxratetop=0;
% % % Maxratebot=0;
% % % for i=col-25:-1:col-500
% % %  if top_ratevalue(i)>Maxratetop
% % %   Maxratetop=top_ratevalue(i);
% % %   Mattateitop=i;
% % %  end
% % %  if bot_ratevalue(i)>Maxratebot
% % %   Maxratebot=bot_ratevalue(i);
% % %   Mattateibot=i;
% % %  end
% % % end
% % % 
% % % imgright=imgconfine;
% % % for j=1:col
% % %  for i=1:row
% % %   if j==Mattateitop
% % %    imgright(i,j)=255;
% % %   end
% % %   if j==Mattateibot
% % %    imgright(i,j)=64;
% % %   end
% % %  end
% % % end
% % % imwrite(imgright,'E:\06右上点.jpg');
% % % figure(8);subplot(111);imshow(imgright);title('imgright');
% % % basePoints = [300,1300;400,1200;440,740;791,120;800,250;850,445];
% % % inputPoints = [150,2600;200,2400;220,1480;396,240;400,500;425,900];
% % % info = imfinfo('D:\study\大三下\tutu\大作业\几何校正图片素材\1.jpg');
% % % imCorrected = imtransform(A,TFORM,'XData',[1,info.Width],'YData',[1,info.Height]);
% % % figure;imshow(A,'InitialMagnification','fit')
% % % figure;
% % % imshow(imCorrected,'InitialMagnification','fit')
% clear all;
% close all;
% clc;
% 
% img= imread('test2.jpg');
% img= rgb2gray(img);
% figure(1);
% imshow(mat2gray(img));
% [M N] = size(img);
% 
% dot=ginput();       %取四个点，依次是左上，右上，左下，右下,这里我取的是书的四个角
% w=round(sqrt((dot(1,1)-dot(2,1))^2+(dot(1,2)-dot(2,2))^2));     %从原四边形获得新矩形宽
% h=round(sqrt((dot(1,1)-dot(3,1))^2+(dot(1,2)-dot(3,2))^2));     %从原四边形获得新矩形高
% 
% y=[dot(1,1) dot(2,1) dot(3,1) dot(4,1)];        %四个原顶点
% x=[dot(1,2) dot(2,2) dot(3,2) dot(4,2)];
% 
% 这里是新的顶点，我取的矩形,也可以做成其他的形状
% 大可以原图像是矩形，新图像是从dot中取得的点组成的任意四边形.:)
% Y=[dot(1,1) dot(1,1) dot(1,1)+h dot(1,1)+h];     
% X=[dot(1,2) dot(1,2)+w dot(1,2) dot(1,2)+w];
% 
% B=[X(1) Y(1) X(2) Y(2) X(3) Y(3) X(4) Y(4)]';   %变换后的四个顶点，方程右边的值
% 联立解方程组，方程的系数
% A=[x(1) y(1) 1 0 0 0 -X(1)*x(1) -X(1)*y(1);             
% 0 0 0 x(1) y(1) 1 -Y(1)*x(1) -Y(1)*y(1);
%    x(2) y(2) 1 0 0 0 -X(2)*x(2) -X(2)*y(2);
% 0 0 0 x(2) y(2) 1 -Y(2)*x(2) -Y(2)*y(2);
%    x(3) y(3) 1 0 0 0 -X(3)*x(3) -X(3)*y(3);
% 0 0 0 x(3) y(3) 1 -Y(3)*x(3) -Y(3)*y(3);
%    x(4) y(4) 1 0 0 0 -X(4)*x(4) -X(4)*y(4);
% 0 0 0 x(4) y(4) 1 -Y(4)*x(4) -Y(4)*y(4)];
% 
% fa=inv(A)*B;        %用四点求得的方程的解，也是全局变换系数
% a=fa(1);b=fa(2);c=fa(3);
% d=fa(4);e=fa(5);f=fa(6);
% g=fa(7);h=fa(8);
% 
% rot=[d e f;
%      a b c;
%      g h 1];        %公式中第一个数是x,Matlab第一个表示y，所以我矩阵1,2行互换了
% 
% pix1=rot*[1 1 1]'/(g*1+h*1+1);  %变换后图像左上点
% pix2=rot*[1 N 1]'/(g*1+h*N+1);  %变换后图像右上点
% pix3=rot*[M 1 1]'/(g*M+h*1+1);  %变换后图像左下点
% pix4=rot*[M N 1]'/(g*M+h*N+1);  %变换后图像右下点
% 
% height=round(max([pix1(1) pix2(1) pix3(1) pix4(1)])-min([pix1(1) pix2(1) pix3(1) pix4(1)]));     %变换后图像的高度
% width=round(max([pix1(2) pix2(2) pix3(2) pix4(2)])-min([pix1(2) pix2(2) pix3(2) pix4(2)]));      %变换后图像的宽度
% imgn=zeros(height,width);
% if min([pix1(1) pix2(1) pix3(1) pix4(1)]) >= 0
%     delta_y = -round(abs(min([pix1(1) pix2(1) pix3(1) pix4(1)])));
% else
%     delta_y = round(abs(min([pix1(1) pix2(1) pix3(1) pix4(1)])));            %取得y方向的负轴超出的偏移量
% end;
% if min([pix1(2) pix2(2) pix3(2) pix4(2)]) >= 0
%     delta_x = -round(abs(min([pix1(2) pix2(2) pix3(2) pix4(2)])));
% else
%     delta_x = round(abs(min([pix1(2) pix2(2) pix3(2) pix4(2)])));            %取得x方向的负轴超出的偏移量
% end;
% delta_y=round(abs(min([pix1(1) pix2(1) pix3(1) pix4(1)])));            %取得y方向的负轴超出的偏移量
% delta_x=round(abs(min([pix1(2) pix2(2) pix3(2) pix4(2)])));            %取得x方向的负轴超出的偏移量
% inv_rot=inv(rot);
% 
% for i = 1-delta_y:height-delta_y                        %从变换图像中反向寻找原图像的点，以免出现空洞，和旋转放大原理一样
%     for j = 1-delta_x:width-delta_x
%         pix=inv_rot*[i j 1]';       %求原图像中坐标，因为[YW XW W]=fa*[y x 1],所以这里求的是[YW XW W],W=gy+hx+1;
%         pix=inv([g*pix(1)-1 h*pix(1);g*pix(2) h*pix(2)-1])*[-pix(1) -pix(2)]'; %相当于解[pix(1)*(gy+hx+1) pix(2)*(gy+hx+1)]=[y x],这样一个方程，求y和x，最后pix=[y x];
%         
%         if pix(1)>=0.5 && pix(2)>=0.5 && pix(1)<=M && pix(2)<=N
%             imgn(i+delta_y,j+delta_x)=img(round(pix(1)),round(pix(2)));     %最邻近插值,也可以用双线性或双立方插值
%         end  
%     end
% end
% 
% figure(2);
% imshow(uint8(imgn));
% result = ocr(imgn); word=result;
% Word={};
% 
% 
% 
% close all;
% clc;
% H=1;                        %索引pix中第一个元素，即高度
% W=2;                        %索引pix中第二个元素，即宽度
% left_right=0.3;               %抬起左边或右边时值为0-1之间，不抬起时为0
% up_down=0;                %抬起上边或下边时值为0-1之间，不抬起时为0
% 
% img=imread('test4.jpg');       %这里v为原图像的高度，u为原图像的宽度
% imshow(img);                    %这里y为变换后图像的高度，x为变换后图像的宽度
% %img=flipud(img);           %注释掉为抬起下边，没注释掉为抬起上边
% %img=fliplr(img);           %注释掉为抬起右边，没注释掉为抬起左边
% [v u]=size(img);
% 
% 
% a=1;b=up_down;c=0;
% d=left_right;e=1;f=0;
% g=up_down/v;h=left_right/u;i=1;
% rot=[a b c;d e f;g h i];
% 
% pix1=[1 1 1]*rot./(g+h+i);                 %变换后图像左上点的坐标
% pix2=[1 u 1]*rot./(g*v+h+i);               %变换后图像右上点的坐标
% pix3=[v 1 1]*rot./(g+h*u+i);               %变换后图像左下点的坐标
% pix4=[v u 1]*rot./(g*v+h*u+i);             %变换后图像右下点的坐标
% 
% height=round(max([abs(pix1(H)-pix3(H))+0.5 abs(pix2(H)-pix3(H))+0.5 ...
%                   abs(pix1(H)-pix4(H))+0.5 abs(pix2(H)-pix4(H))+0.5]));     %变换后图像的高度
% 
% width=round(max([abs(pix1(W)-pix2(W))+0.5 abs(pix3(W)-pix2(W))+0.5 ...
%                  abs(pix1(W)-pix4(W))+0.5 abs(pix3(W)-pix4(W))+0.5]));      %变换后图像的宽度
% imgn=zeros(height,width);
% 
% delta_y=abs(min([pix1(H)-0.5 pix2(H)-0.5 pix3(H)-0.5 pix4(H)-0.5]));            %取得y方向的负轴超出的偏移量
% delta_x=abs(min([pix1(W)-0.5 pix2(W)-0.5 pix3(W)-0.5 pix4(W)-0.5]));            %取得x方向的负轴超出的偏移量
% 
% for y=1-floor(delta_y):height-floor(delta_y)
%     for x=1-floor(delta_x):width-floor(delta_x)
%         pix=[y x 1]/rot*(g*y+h*x+i);                                %用变换后图像的点的坐标去寻找原图像点的坐标，                                         
%                                                             %否则有些变换后的图像的像素点无法完全填充
%         if pix(H)>=0.5 && pix(W)>=0.5 && pix(H)<=v && pix(W)<=u
%             imgn(y+floor(delta_y),x+floor(delta_x))=img(round(pix(H)),round(pix(W)));
%         end   
%         
%     end
% end
% figure,imshow(uint8(imgn));
% 
% %%二次变换
% img=imgn;
% [v u]=size(img);
% a=1;b=-b/2;c=0;
% d=-d/2;e=1;f=0;
% g=0;h=0;i=1;
% rot=[a b c;d e f;g h i];
% 
% pix1=[1 1 1]*rot./(g+h+i);                 %变换后图像左上点的坐标
% pix2=[1 u 1]*rot./(g*v+h+i);               %变换后图像右上点的坐标
% pix3=[v 1 1]*rot./(g+h*u+i);               %变换后图像左下点的坐标
% pix4=[v u 1]*rot./(g*v+h*u+i);             %变换后图像右下点的坐标
% 
% height=round(max([abs(pix1(H)-pix3(H))+0.5 abs(pix2(H)-pix3(H))+0.5 ...
%                   abs(pix1(H)-pix4(H))+0.5 abs(pix2(H)-pix4(H))+0.5]));     %变换后图像的高度
% 
% width=round(max([abs(pix1(W)-pix2(W))+0.5 abs(pix3(W)-pix2(W))+0.5 ...
%                  abs(pix1(W)-pix4(W))+0.5 abs(pix3(W)-pix4(W))+0.5]));      %变换后图像的宽度
% imgn=zeros(height,width);
% 
% delta_y=abs(min([pix1(H)-0.5 pix2(H)-0.5 pix3(H)-0.5 pix4(H)-0.5]));            %取得y方向的负轴超出的偏移量
% delta_x=abs(min([pix1(W)-0.5 pix2(W)-0.5 pix3(W)-0.5 pix4(W)-0.5]));            %取得x方向的负轴超出的偏移量
% 
% for y=1-floor(delta_y):height-floor(delta_y)
%     for x=1-floor(delta_x):width-floor(delta_x)
%         pix=[y x 1]/rot*(g*y+h*x+i);                                %用变换后图像的点的坐标去寻找原图像点的坐标，                                         
%                                                             %否则有些变换后的图像的像素点无法完全填充
%         if pix(H)>=0.5 && pix(W)>=0.5 && pix(H)<=v && pix(W)<=u
%             imgn(y+floor(delta_y),x+floor(delta_x))=img(round(pix(H)),round(pix(W)));
%         end   
%         
%     end
% end
% %imgn=flipud(imgn);             %注释掉为抬起下边，没注释掉为抬起上边
% %imgn=fliplr(imgn);             %注释掉为抬起右边，没注释掉为抬起左边
% figure,imshow(uint8(imgn));















% clc;
% close all;
% clear;
% %% 一、图像预处理
% % imgori = rgb2gray(im2double(imread('picture.jpg')));
% imgori =imread('picture.jpg');
% figure(1);imshow(imgori);title('原图');
% %同态滤波
% img_gray=rgb2gray(imgori);
% I=im2double(img_gray); 
% [M,N]=size(I);
% P = 2*M;
% Q = 2*N; 
% I=log(I+1);
% FI=fft2(I,P,Q);    %傅里叶变换，并扩大尺寸填充0
% % 滤波器
% rL=0.7;    %0.7
% rH=0.9;   %0.9
% c=3;       %锐化参数5
% D0=50;  %100
% [v, u] = meshgrid(1:Q, 1:P);
% u = u - floor(P/2); 
% v = v - floor(Q/2); 
% D = u.^2 + v.^2;  % 距离的平方
% H = 1-exp(-c*(D./D0^2));    %高斯滤波器
% H = (rH - rL)*H + rL;    
% H=ifftshift(H);  %对H做反中心化
% I3=real(ifft2(H.*FI));  %傅里叶逆变换
% I4=I3(1:M, 1:N);  %截取一部分
% img_tong=exp(I4)-1;  %取指数
% figure(2);
% imshow(img_tong);
% img = medfilt2(img_tong);
% figure(3);
% imshow(img);
% % figure(1);imshow(imgori);title('原图');
% % img = medfilt2(imgori);
% % figure(3);
% % imshow(img);
% %% 1.提取文字大区域部分
% level = graythresh(img);    
% A = imcomplement(im2bw(img,level));  %大津阈值分割后图像取反
% A = medfilt2(A,[3 3]);  %中值滤波
% %figure;imshow(A);
% [height,width] = size(A);
% %初步去除图片上非文字部分（如阴影等）
% th = height*width/1000;    %去除大阴影的阈值
% CC = bwconncomp(A,8);
% area = cellfun(@numel,CC.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% idx = (area>=th);        %去除非文字部分
% idx0 = find(idx == 1);         %得到第几个目标是大目标
% [h,w] = size(idx0);
% for i = 1:w
%     A(CC.PixelIdxList{idx0(i)}) = 0;    %每个细小目标值赋为0
% end
% yh=height*width/100000;
% idx_0=(area<yh);
% idx_00=find(idx_0==1);
% [a,b]=size(idx_00);
% for j=1:b
%     A(CC.PixelIdxList{idx_00(j)}) = 0; 
% end
% figure(4);imshow(A)
% 
% 
% % MN=[4,40];
% % se=strel('rectangle',MN);
% % B=imdilate(A,se);%图像A1被结构元素B膨胀，背景置0，文字置1
% se = strel('disk',45); %膨胀运算
% B = imdilate(A,se); 
% figure(5);imshow(B);
% %进一步去除非文字部分的小目标
% CC2 = bwconncomp(B);
% numPixels2 = cellfun(@numel,CC2.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% max2 = max(numPixels2);        %得到文字区域（像素最多的连通域）
% position = find(numPixels2 == max2); 
% renew_img = zeros(height,width);
% renew_img(CC2.PixelIdxList{position}) = A(CC2.PixelIdxList{position});    %将文字区域的图像提取出来
% figure(6);imshow(renew_img);
% % CC5 = bwconncomp(renew_img);
% % numPixels = cellfun(@numel,CC5.PixelIdxList);  %像素统计（每个连通区域的像素个数）
% % max3 = max(numPixels);
% % position = find(numPixels == max3); 
% % renew_img(CC5.PixelIdxList{position}) = 0;
% % figure(7);imshow(renew_img);
% %% 标记每行文字
% se = strel('rectangle',[5 45]); %膨胀运算
% C = imdilate(renew_img,se); 
% L = bwconncomp(C);   %更新连通域
% %   stats = regionprops('table',bw,'Centroid', ...
% %                              'MajorAxisLength','MinorAxisLength');
% %   
% %           % Get centers and radii of the circles
% %           centers = stats.Centroid;
% %           diameters = mean([stats.MajorAxisLength stats.MinorAxisLength],2);
% %           radii = diameters/2;
% %   
% %           % Plot the circles
% %           hold on
% %           viscircles(centers,radii);
% %           hold off
% stats = regionprops(L,'centroid');   %STATS = regionprops(L,properties)该函数用来测量标注矩阵L中每一个标注区域的一系列属性。获得中心点
% centroids = cat(1,stats.Centroid);   %cat连接数组
% figure(8);imshow(C);hold on;plot(centroids(:,1),centroids(:,2),'g*');hold off
% [L,num] = bwlabel(C);  %num代表有几行文字
% %% 计算偏移量
% D = C;
% for i=1:5:width   %区域分割细度
%     D(:,i) = 0;
% end
% CC_new = bwconncomp(D);
% stats_new = regionprops(CC_new,'centroid');   %获得中心点
% centroids = cat(1,stats_new.Centroid);
% [hc,wc] = size(centroids);  %hc代表有多少个中心点
% centr_inf = zeros(hc,wc+2);    %wc=2,第三列是位于第几行的信息
% centr_inf(1:hc,1:2) = centroids;
% for i = 1:hc
%     for j = 1:num
%         if (L(round(centroids(i,2)),round(centroids(i,1))) == j)   %L是每行信息
%             centr_inf(i,3) = j;
%            % fprintf('i=%d,j=%d\n',i,j);
%         end
%     end
% end
% %% 存储每个中心点的偏移量
% for i = 1:num
%     pos = find(centr_inf(:,3) == i);
%     centr_inf(pos,4) = mean(centr_inf(pos,2))-centr_inf(pos,2);    %存储偏移量
%     a = centr_inf(pos,4);
% %    id1 = find(a>=mean(a));     %分段平滑
% %    id2 = find(a<mean(a));
% %    a(id1) = smoothts(a(id1),'e',100);
% %    a(id2) = smoothts(a(id2),'e',100);
% %    a(id2) = medfilt1(a(id2),50);
% 
%     a = smoothts(a,'e',100);  %单独平滑
%     a = medfilt1(a,50);
%     centr_inf(pos,4) = a;
%     fprintf('%d\n',i);
% end
% figure(9);imshow(D);hold on;plot(centr_inf(:,1),centr_inf(:,2),'g*');hold off
% %% 非线性畸变校正
% finalimg = zeros(height,width);
% [L_D,num_D] = bwlabel(D);
% STATS2 = regionprops(L_D,'PixelList');%PixelList 存储的坐标是列在前，行在后
% mask = zeros(height,width);
% for i = 1:num_D
%     [h,w] = size(STATS2(i).PixelList);
%     %maxf = max(STATS2(i).PixelList(:,1));
%     %len = length(find(STATS2(i).PixelList(:,1) == maxf));
%     statsnow = [STATS2(i).PixelList;[STATS2(i).PixelList(1:h,1)+1,STATS2(i).PixelList((1:h),2)];[STATS2(i).PixelList(1:h,1)+2,STATS2(i).PixelList((1:h),2)];[STATS2(i).PixelList(1:h,1)+3,STATS2(i).PixelList((1:h),2)];[STATS2(i).PixelList(1:h,1)+4,STATS2(i).PixelList((1:h),2)]];
%    %[h,w] = size(statsnow);
%     %for j = 1:h
%         finalimg(statsnow(:,2)+round(centr_inf(i,4)),statsnow(:,1)) = renew_img(statsnow(:,2),statsnow(:,1));%行，列
%    % end
% end
% figure;imshow(imcomplement(finalimg));
% imwrite(imcomplement(finalimg),'myphoto.jpg')
