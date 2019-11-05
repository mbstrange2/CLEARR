% Plot color maps for each readout configuration

nwire = 16;
MEA = zeros(16,32);

for i = 1:nwire
    MEA(i:nwire:end) = (256/nwire - 1) * i;
end
figure(1)
% Display it.
image(MEA);
% Initialize a color map array of 256 colors.
colorMap = jet(256);
% Apply the colormap and show the colorbar
colormap(colorMap);
