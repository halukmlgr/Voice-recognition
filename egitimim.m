%%201713709057 Haluk MALGIR
%% Veri Setinin Eklenmesi

 ADS = audioDatastore('Egitim','IncludeSubfolders',1);
 ADS.Labels = extractBetween(ADS.Files,fullfile('Egitim',filesep),filesep);

 
%% Veri setini %80 Train ve %20 Test olarak ikiye ayirmak
[ADSTrain,ADSTest] = splitEachLabel(ADS,0.8);
[audioIn,dsInfo] = read(ADSTrain);
audioIn=[audioIn(:,1)];
Fs = dsInfo.SampleRate;
t = (1/Fs)*(0:length(audioIn)-1);

%% Pointer'i sifirliyoruz.
reset(ADSTrain)

%% 40ms arayla Ses dosyasini 200ms lik parcalara bolerek sessiz parcalardan kurtariyoruz.
frameDuration = 200e-3;
overlapDuration = 40e-3;
frameLength = floor(Fs*frameDuration); 
overlapLength = round(Fs*overlapDuration);
[XTrain,YTrain] = preprocessAudioData(ADSTrain,frameLength,overlapLength,Fs);
[XTest,YTest] = preprocessAudioData(ADSTest,frameLength,overlapLength,Fs);
%% STANDART CNN
numFilters = 80;
filterLength =5;
numSpeakers = numel(unique(ADS.Labels));
layers = [ 
    imageInputLayer([1 frameLength 1])
    
    % İlk evrişim katmanı
    convolution2dLayer([1 filterLength],numFilters)
    batchNormalizationLayer
    leakyReluLayer(0.2)
    maxPooling2dLayer([1 3])
    
    % Bu katmanı 2 evrişim katmanı takip eder.
    
    convolution2dLayer([1 5],60)
    batchNormalizationLayer
    leakyReluLayer(0.2)
    maxPooling2dLayer([1 3])
    
    convolution2dLayer([1 5],60)
    batchNormalizationLayer
    leakyReluLayer(0.2)
    maxPooling2dLayer([1 3])

    % Bunu 3 tam bağlı katman takip eder.
    
    fullyConnectedLayer(256)
    batchNormalizationLayer
    leakyReluLayer(0.2)
    
    fullyConnectedLayer(256)
    batchNormalizationLayer
    leakyReluLayer(0.2)

    fullyConnectedLayer(256)
    batchNormalizationLayer
    leakyReluLayer(0.2)

    fullyConnectedLayer(numSpeakers)
    softmaxLayer
    classificationLayer
    ];
%%

%%
numEpochs =15;
miniBatchSize =128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions("adam", ...
    "Shuffle","every-epoch", ...
    "MiniBatchSize",miniBatchSize, ...
    "Plots","training-progress", ...
    "Verbose",false,"MaxEpochs",numEpochs, ...
    "ValidationData",{XTest,categorical(YTest)}, ...
    "ValidationFrequency",validationFrequency, ...
    'ExecutionEnvironment', 'cpu');

%% AGI EGITMEK
net = trainNetwork(XTrain,YTrain,layers,options);
%% Test verilerinin tahmin sonuclari
YPred = classify(net,XTest);
accuracy = sum(YPred == YTest)/numel(YTest);
%% Support Function
Egitilmis = net;
 save Egitilmis
function [X,Y] = preprocessAudioData(ADS,SL,OL,Fs)
    if ~isempty(ver('parallel'))
        pool = gcp;
        numPar = numpartitions(ADS,pool);
    else
        numPar = 1;
    end
    parfor ii = 1:numPar
    
        X = zeros(1,SL,1,0);
        Y = zeros(0);
        subADS = partition(ADS,numPar,ii);
        
        while hasdata(subADS)
            [audioIn,dsInfo] = read(subADS);
            audioIn=[audioIn(:,1)];
          
            
            speechIdx = detectSpeech(audioIn,Fs);
            numChunks = size(speechIdx,1);
            audioData = zeros(1,SL,1,0);      
            
            for chunk = 1:numChunks
                % Isaretlenen sesler temizleniyor
                audio_chunk = audioIn(speechIdx(chunk,1):speechIdx(chunk,2));
                audio_chunk = buffer(audio_chunk,SL,OL);
                q = size(audio_chunk,2);
                
                % Ses 200 ms'lik parçalara ayriliyor
                audio_chunk = reshape(audio_chunk,1,SL,1,q);
                
                % Mevcut sesle birleştir
                audioData = cat(4,audioData,audio_chunk);
            end
            
            audioLabel = str2double(dsInfo.Label{1});
            
            % Matrisi çoğaltarak eğitim ve test için etiketler oluşturun
            audioLabelsTrain = repmat(audioLabel,1,size(audioData,4));
            
            % Mevcut konuşmacı için veri isaretleniyor
            X = cat(4,X,audioData);
            Y = cat(2,Y,audioLabelsTrain);
        end
            
        XC{ii} = X;
        YC{ii} = Y;
    end
    
    X = cat(4,XC{:});
    Y = cat(2,YC{:});
    
    Y = categorical(Y);
    
end