% Taku Ito
% Analyzing Siegel et al. 2015 data set 
% 08/10/2018

%clear all; close all;
datadir = '/projects3/NHPActFlow/data/';
%sessionNames = {'100706','100730','100804','100820','100827','100913','100921','101024','101122','101128','101207','101217','110110_01','110115_01','100724','100731','100817','100823','100828','100915','101008','101027','101123','101202','101209','110106','110110_02','110120','100725','100802','100818','100824','100907','100917','101009','101028','101124','101203','101210','110107_01','110111_01','110121','100726','100803','100819','100826','100910','100920','101023','101030','101127','101206','101216','110107_02','110111_02.mat'};

sessionNames = {'101009','101028','101124','101203','101210','110107_01','110111_01','110121','100726','100803','100819','100826','100910','100920','101023','101030','101127','101206','101216','110107_02','110111_02'};
sessionNames = {'100724','100731','100817','100823','100828','100915','101008','101027','101123','101202','101209','110106','110110_02','110120','100725','100802','100818','100824','100907','100917'};
%#sessionNames = {'100706','100730','100804','100820','100827','100913','100921','101024','101122','101128','101207','101217','110110_01'};


for i=1:length(sessionNames)
    binsize = 1; % ms
    datadir = '/projects3/NHPActFlow/data/';
    %sessionNames = {'100724','100731','100817','100823','100828','100915','101008','101027','101123','101202','101209','110106','110110_02','110120','100725','100802','100818','100824','100907','100917'};
    %sessionNames = {'101009','101028','101124','101203','101210','110107_01','110111_01','110121','100726','100803','100819','100826','100910','100920','101023','101030','101127','101206','101216','110107_02','110111_02'};
    %sessionNames = {'100706','100730','100804','100820','100827','100913','100921','101024','101122','101128','101207','101217','110110_01'};
    sessionNames = {'110115_01'};
    
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
    nAreas = length(unique(unitInfo.area));
    uniqueAreas = unique(unitInfo.area);
    nCells = size(spikeTimes,2);
    nTrials = size(spikeTimes,1);
    sta = zeros(nAreas,nBins,nTrials);
    cellsperarea = size(nAreas,1);
    

    for trial=1:nTrials
        %disp(['Running analysis for trial ' num2str(trial)])

        tmp_parfor{trial} = zeros(nAreas,nBins);
        for area=1:nAreas
            cells = find(ismember(unitInfo.area,uniqueAreas(area)));
            for cellcount=1:length(cells)
                cell = cells(cellcount);
                if ~isempty(spikeTimes{trial,cell})
                    nSpikes = length(spikeTimes{trial,cell});
                    % Place spike in sta array
                    for spike=1:nSpikes
                        spikeTime = int16(spikeTimes{trial,cell}(spike) * 1000); % convert to ms
                        % Find index
                        timeInd = min(find(timeID>spikeTime));
                        %%sta(cell,timeInd,trial) = 1;
                        tmp_parfor{trial}(area,timeInd) = tmp_parfor{trial}(area,timeInd) + 1;
                    end
                end
            end

        end
    end

    % Create proper array
    for trial=1:nTrials
        sta(:,:,trial) = tmp_parfor{trial};
    end

    %% New
    % Remove first element of matrix (neurons that don't belong anywhere)
    sta(1,:,:) = [];
    uniqueAreas(1) = [];
    nAreas = nAreas-1;
    %% End

    h5create(['/projects3/TaskFCMech/data/nhpData/' sessionNames{i} '_perArea_v2.h5'], '/sta', [nAreas,nBins,nTrials])
    h5write(['/projects3/TaskFCMech/data/nhpData/' sessionNames{i} '_perArea_v2.h5'],'/sta',sta)
    
    % Save task data
    %writetable(trialInfo.allTasks,['/projects3/TaskFCMech/data/nhpData/' sessionNames{i} '_trialInfoAllTasks.csv']);
    %writetable(trialInfo.molcol,['/projects3/TaskFCMech/data/nhpData/' sessionNames{i} '_trialInfoMOCOL.csv']);
    
    % Save unit (neuron) data
    tmptable = cell2table(unique(unitInfo.area));
    writetable(tmptable,['/projects3/TaskFCMech/data/nhpData/' sessionNames{i} '_areaIndices_v2.csv']);

    clear all;
    

end

    
