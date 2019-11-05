function [neuralChannel] = neuralDataAveraging(neuralChannel, sigma)

for i = 2:size(neuralChannel,1)-1
    if(neuralChannel(i) == 0)
        neuralChannel(i) = round((neuralChannel(i-1) + neuralChannel(i+1))/2);
    end
    if(abs(neuralChannel(i-1)) < 4)
        neuralChannel(i) = neuralChannel(i) + randi([-sigma sigma]);
    end
end

