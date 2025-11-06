% Create the processor configuration object
hPC = dlhdl.ProcessorConfig();
hPC.TargetFrequency=150;

%% Define board
hPC.TargetPlatform = 'Intel Arria 10 SoC development kit';

% Add the synthesis tool to the MATLAB path. Uncomment the line below and
% provide the correct path.
% quartus_path = strcat(getenv("QUARTUS_ROOTDIR"), "/bin/quartus");
quartus_path = '/opt/FPGA/Intel/intelFPGA/21.1std/quartus/bin/quartus'

hdlsetuptoolpath('ToolName', 'Altera Quartus II','ToolPath', quartus_path);

% Create workflow config object
hWC = hdlcoder.WorkflowConfig('SynthesisTool', hPC.SynthesisTool,'TargetWorkflow','Deep Learning Processor');
hWC.RunTaskGenerateRTLCodeAndIPCore = true;
hWC.RunTaskCreateProject = true;
hWC.RunTaskBuildFPGABitstream = true; % Changed to True

% Call dlhdl.buildProcessor. This generates HDL code and creates the
% Quartus project.
dlhdl.buildProcessor(hPC, 'WorkflowConfig', hWC, 'OverrideResourceCheck', true);
hPC.estimateResources;