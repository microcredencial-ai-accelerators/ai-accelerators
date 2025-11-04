function hB = plugin_board()
% Board definition for Terasic DE10-Nano SoC

% Construct board object
hB = hdlcoder.Board;

hB.BoardName = 'Terasic DE10-Nano SoC';

% FPGA device information
hB.FPGAVendor  = 'Altera';
hB.FPGAFamily  = 'Cyclone V';
hB.FPGADevice  = '5CSEBA6U23I7';
hB.FPGAPackage = '';
hB.FPGASpeed   = '';

% Tool information
hB.SupportedTool = {'Altera QUARTUS II'};

% JTAG chain position
hB.JTAGChainPosition = 2;

% Size of external DDR memory in bytes (1 GB DDR3 on DE10-Nano)
hB.ExternalMemorySize = 0x40000000;  % 1 GB

%% Add Interfaces

% Default I/O standard
defaultIOStandard = 'IO_STANDARD = 3.3-V LVTTL';

% General-purpose LEDs (8 red LEDs on DE10-Nano)
hB.addExternalIOInterface( ...
    'InterfaceID',    'LEDs', ...
    'InterfaceType',  'OUT', ...
    'PortName',       'LED', ...
    'PortWidth',      8, ...
    'FPGAPin',        {'PIN_W15', 'PIN_AA24', 'PIN_V16', 'PIN_V15', ...
                       'PIN_AF26', 'PIN_AE26', 'PIN_Y16', 'PIN_AA23'}, ...
    'IOPadConstraint', {defaultIOStandard});

% Push button (KEY0 and KEY1)
hB.addExternalIOInterface( ...
    'InterfaceID',    'User Push Button 0', ...
    'InterfaceType',  'IN', ...
    'PortName',       'KEY', ...
    'PortWidth',      2, ...
    'FPGAPin',        {'PIN_H17', 'PIN_AH16'}, ...
    'IOPadConstraint', {defaultIOStandard});

end