function [dataIn] = neuralDataAddNoise(dataIn, sigma)

for k = 2:size(dataIn,2)
    for i = 2:size(dataIn,1)-1
        if(abs(dataIn(i-1,k)) < 4)
            dataIn(i,k) = dataIn(i,k) + randi([-sigma sigma]);
        end
    end
end
