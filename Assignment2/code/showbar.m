function showbar(bar)
[sv,sh] = size(bar);
if (ceil(sh/2)*2~=sh)
    error('input must be of size [1, M], with M being even');
end
%if sv>

if min(bar)==0
    bar  = 2*bar - 1;
end
nbars   = size(bar,1);
n       = ceil(sqrt(nbars));
figure
for k=1:nbars
    bar_top = bar(k,1:end/2);
    bar_bot = bar(k,end/2+1:end);
    hf=subplot(n,n,k);
   % set(hf,'colormap', [0 0 0; 1 1 1]);
    hinton([bar_top;bar_bot]);
end

function h = hinton(w);
%HINTON	Plot Hinton diagram for a weight matrix.
%
%	Description
%
%	HINTON(W) takes a matrix W and plots the Hinton diagram.
%
%	H = HINTON(NET) also returns the figure handle H which can be used,
%	for instance, to delete the  figure when it is no longer needed.
%
%	To print the figure correctly in black and white, you should call
%	SET(H, 'INVERTHARDCOPY', 'OFF') before printing.
%
%	See also
%	DEMHINT, HINTMAT, MLPHINT
%

%	Copyright (c) Ian T Nabney (1996-2001)

% Set scale to be up to 0.9 of maximum absolute weight value, where scale
% defined so that area of box proportional to weight value.

% Use no more than 640x480 pixels
xmax = 640; ymax = 480;

% Offset bottom left hand corner
x01 = 40; y01 = 40;
x02 = 80; y02 = 80;

% Need to allow 5 pixels border for window frame: but 30 at top
border = 5;
top_border = 30;

ymax = ymax - top_border;
xmax = xmax - border;

% First layer

[xvals, yvals, color] = hintmat(w);
% Try to preserve aspect ratio approximately
if (8*size(w, 1) < 6*size(w, 2))
    delx = xmax; dely = xmax*size(w, 1)/(size(w, 2));
else
    delx = ymax*size(w, 2)/size(w, 1); dely = ymax;
end
if 0
h = figure('Color', [0.5 0.5 0.5], ...
    'Name', 'Hinton diagram', ...
    'NumberTitle', 'off', ...
    'Colormap', [0 0 0; 1 1 1], ...
    'Units', 'pixels', ...
    'Position', [x01 y01 delx dely]);
end
set(gcf,'Color', [0.5 0.5 0.5], 'Colormap', [0 0 0; 1 1 1]);
set(gca,'visible','off')
%set(gca, 'Visible', 'off', 'Position', [0 0 1 1]);
hold on
ha=patch(xvals', yvals', color', 'Edgecolor', 'none');
%set(ha,'visible','off')
ha = axes; set(ha,'visible','off');
axis equal;

function [xvals, yvals, color] = hintmat(w);
%HINTMAT Evaluates the coordinates of the patches for a Hinton diagram.
%
%	Description
%	[xvals, yvals, color] = hintmat(w)
%	  takes a matrix W and returns coordinates XVALS, YVALS for the
%	patches comrising the Hinton diagram, together with a vector COLOR
%	labelling the color (black or white) of the corresponding elements
%	according to their sign.
%
%	See also
%	HINTON
%

%	Copyright (c) Ian T Nabney (1996-2001)

% Set scale to be up to 0.9 of maximum absolute weight value, where scale
% defined so that area of box proportional to weight value.

w = flipud(w);
[nrows, ncols] = size(w);

scale = 0.45*sqrt(abs(w)/max(max(abs(w))));
scale = scale(:);
color = 0.5*((2*double(w(:)>0)-1) + 3);

delx = 1;
dely = 1;
[X, Y] = meshgrid(0.5*delx:delx:(ncols-0.5*delx), 0.5*dely:dely:(nrows-0.5*dely));

% Now convert from matrix format to column vector format, and then duplicate
% columns with appropriate offsets determined by normalized weight magnitudes.

xtemp = X(:);
ytemp = Y(:);

xvals = [xtemp-delx*scale, xtemp+delx*scale, ...
    xtemp+delx*scale, xtemp-delx*scale];
yvals = [ytemp-dely*scale, ytemp-dely*scale, ...
    ytemp+dely*scale, ytemp+dely*scale];


