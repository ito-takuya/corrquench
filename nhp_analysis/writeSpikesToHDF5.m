% Taku Ito
% Analyzing Siegel et al. 2015 data set 
% 08/10/2018

%clear all; close all;
datadir = '/projects3/NHPActFlow/data/';
%sessionNames = {'100706','100730','100804','100820','100827','100913','100921','101024','101122','101128','101207','101217','110110_01','110115_01','100724','100731','100817','100823','100828','100915','101008','101027','101123','101202','101209','110106','110110_02','110120','100725','100802','100818','100824','100907','100917','101009','101028','101124','101203','101210','110107_01','110111_01','110121','100726','100803','100819','100826','100910','100920','101023','101030','101127','101206','101216','110107_02','110111_02.mat'};

sessionNames = {'110115_01','100724','100731','100817','100823','100828','100915','101008','101027','101123','101202','101209','110106','110110_02','110120','100725','100802','100818','100824','100907','100917','101009','101028','101124','101203','101210','110107_01','110111_01','110121','100726','100803','100819','100826','100910','100920','101023','101030','101127','101206','101216','110107_02','110111_02'};
sessionNames = {'101216','110107_02','110111_02'};
% '101128',

%sessionNames = {'100724'};

for i=1:length(sessionNames)
    binsize = 1; % ms
    datadir = '/projects3/NHPActFlow/data/';
    sessionNames = {'101216','110107_02','110111_02'};
    
    %% First load in session
    disp(['Loading session ' sessionNames{i}])
    disp(['Session ' num2str(i) ' out of ' num2str(length(sessionNames))])
    load([datadir sessionNames{i}]);
    close all

    %% 
    % Measure the STA for each neuron in this session
    tmin = -4000; %ms
    tmax = 4000; %ms
%    tmin = -2500; %ms
%    tmax = 3500; %ms
    timeID = tmin:binsize:tmax;
    nBins = length(timeID);
    nCells = size(spikeTimes,2);
    nTrials = size(spikeTimes,1);
    sta = zeros(nCells,nBins,nTrials);

    for trial=1:nTrials
        %disp(['Running analysis for trial ' num2str(trial)])

        tmp_parfor{trial} = zeros(nCells,nBins);
        for cell=1:nCells
            if ~isempty(spikeTimes{trial,cell})
                nSpikes = length(spikeTimes{trial,cell});
                % Place spike in sta array
                for spike=1:nSpikes
                    spikeTime = int16(spikeTimes{trial,cell}(spike) * 1000); % convert to ms
                    % Find index
                    timeInd = min(find(timeID>spikeTime));
                    %%sta(cell,timeInd,trial) = 1;
                    tmp_parfor{trial}(cell,timeInd) = 1;
                end
            end
        end
    end

    % Create proper array
    for trial=1:nTrials
        sta(:,:,trial) = tmp_parfor{trial};
    end

    h5create(['/projects3/TaskFCMech/data/nhpData/' sessionNames{i} '.h5'], '/sta', [nCells,nBins,nTrials])
    h5write(['/projects3/TaskFCMech/data/nhpData/' sessionNames{i} '.h5'],'/sta',sta)
    
    % Save task data
    writetable(trialInfo.allTasks,['/projects3/TaskFCMech/data/nhpData/' sessionNames{i} '_trialInfoAllTasks.csv']);
    
    % Save unit (neuron) data
    writetable(unitInfo,['/projects3/TaskFCMech/data/nhpData/' sessionNames{i} '_unitInfo.csv']);

    clear all;
    

end

    
