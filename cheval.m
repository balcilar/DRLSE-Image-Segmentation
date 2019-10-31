function cheval_segmentation
% This code was written by Muhammet Balcilar , France,
% muhammetbalcilar@gmail.com, inspired from following reference
%
%  This Matlab code implements an edge-based active contour model as an
%  application of the Distance Regularized Level Set Evolution (DRLSE) formulation in Li et al's paper:
%
%      C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation", 
%        IEEE Trans. Image Processing, vol. 19 (12), pp.3243-3254, 2010.

close all

%% step1, read grayscale image
Img=imread('Inputs/out_meanshift_big.jpg');
Img=rgb2gray(Img);
frm=0;
%% step2, set params
timestep=30;  % time step
mu=0.2;  % coefficient of the distance regularization term R(phi)
lambda=5; %coefficient of the weighted length term L(phi)
alfa= -3;  %  coefficient of the weighted area term A(phi)
epsilon=1.5; % papramater that specifies the width of the DiracDelta function
c0=2;
maxiter=220;
sigma=2.0;    % scale parameter in Gaussian kernel

%% step3 smooth image with gaussian filter
G=fspecial('gaussian',30,sigma); % 15 Caussian kernel
Img_smooth=conv2(Img,G,'same');  % smooth image by Gaussiin convolution
figure(1);
imagesc(Img_smooth,[0, 255]); axis off; axis equal; colormap(gray);
title('Smoothed image');

%% step4 calculate edge indicator according to Eq23
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
%load hednms
%f=double(E);
g=1./(1+f);  % edge indicator function.
g=exp(-f);
figure(2);
imagesc(g); axis off; axis equal; 
title('g,  edge indicator');

%% step5, set initial phi

im=imread('Inputs/out_meanshift_big_draw4.jpg');
bw=im(:,:,1)>220;
L=bwlabel(bw);
ul=unique(L(:));
a=hist(L(:),ul);
a=sortrows([a' ul],-1);
bw=L==a(2,2);

bww=imfill(bw, 'holes')-bw;

L=bwlabel(bww);
ul=unique(L(:));
a=hist(L(:),ul);
a=sortrows([a' ul],-1);
bw=L==a(2,2);

%bw=zeros(size(bw));
%bw(510:1060,280:950)=1;

phi = -c0*ones(size(Img));
phi(bw==1)=c0;
%phi(350:370,200:220)=-c0;
figure(3);
imagesc(phi);
axis off; axis equal;colormap(jet);
title('initial phi matrix');

[vx, vy]=gradient(g);
figure(4);
subplot(1,2,1);imagesc(vx); title('x directioned gradient of g');
subplot(1,2,2);imagesc(vy); title('y directioned gradient of g');

for k=1:maxiter
    
    if mod(k,1)==0
        frm=frm+1;
        %close all;
        h=figure(5);
        set(gcf,'color','w');
        %subplot(1,2,1);
        II=Img;
        II(:,:,2)=Img;II(:,:,3)=Img;
        imshow(II); axis off; axis equal; hold on;  
        q=contour(phi, [0,0], 'r');
        msg=['contour result , iteration number=' num2str(k)];
        title(msg);

        
        frame = getframe(h);
        im = frame2im(frame);
        [imind,cm] = rgb2ind(im,256);
        %Write to the GIF File
        if frm == 1        
            imwrite(imind,cm,'Outputs/cheval2.gif','gif', 'Loopcount',inf);
        else        
            imwrite(imind,cm,'Outputs/cheval2.gif','gif','WriteMode','append');
        end
    end
    
    
    %% step6, check boundary conditions
    phi=NeumannBoundCond(phi);
    
    %% step 7 calculate differential of regularized term in Eq.30
    distRegTerm=distReg_p2(phi);
    
    %% step8 calculate differential of area term in Eq.30
    diracPhi=Dirac(phi,epsilon);
    areaTerm=diracPhi.*g;
    
    %% step9 calculate differential of length term in Eq.30
    [phi_x,phi_y]=gradient(phi);
    s=sqrt(phi_x.^2 + phi_y.^2);
    Nx=phi_x./(s+1e-10); % add a small positive number to avoid division by zero
    Ny=phi_y./(s+1e-10);
    edgeTerm=diracPhi.*(vx.*Nx+vy.*Ny) + diracPhi.*g.*div(Nx,Ny);
    
    %% step 10 update phi according to Eq.20
    phi=phi + timestep*(mu/timestep*distRegTerm + lambda*edgeTerm + alfa*areaTerm);
    
    %% show result in every 50 iteration
    
    
    
    %% step 11 if maxiter done then finish, else return step6
end
%% Step 12. show last iteration results
figure(6);
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); hold on;  contour(phi, [0,0], 'r');
msg=['phi result , iteration number=' num2str(k)];
title(msg);


function f = distReg_p2(phi)
% compute the distance regularization term with the double-well potential p2 in eqaution (16)
[phi_x,phi_y]=gradient(phi);
s=sqrt(phi_x.^2 + phi_y.^2);
a=(s>=0) & (s<=1);
b=(s>1);
ps=a.*sin(2*pi*s)/(2*pi)+b.*(s-1);  % compute first order derivative of the double-well potential p2 in eqaution (16)
dps=((ps~=0).*ps+(ps==0))./((s~=0).*s+(s==0));  % compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
f = div(dps.*phi_x - phi_x, dps.*phi_y - phi_y) + 4*del2(phi);

function f = div(nx,ny)
[nxx,junk]=gradient(nx);
[junk,nyy]=gradient(ny);
f=nxx+nyy;

function f = Dirac(x, sigma)
f=(1/2/sigma)*(1+cos(pi*x/sigma));
b = (x<=sigma) & (x>=-sigma);
f = f.*b;

function g = NeumannBoundCond(f)
% Make a function satisfy Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);