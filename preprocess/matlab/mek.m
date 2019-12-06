clear all; clc;

dataSet = '2015-11-09-3/';
compressed_data = 1;
for base_time_min=20:5:25
    run getBinToMatFile_spike_Cell.m;
end
