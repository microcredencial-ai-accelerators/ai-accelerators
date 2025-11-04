function [boardList, workflow] = hdlcoder_board_customization
% Board plugin registration file for Terasic DE10-Nano SoC
% 1. Any registration file with this name on MATLAB path will be picked up
% 2. Registration file returns a cell array pointing to the location of 
%    the board plugin(s)
% 3. Board plugin must be a package folder accessible from MATLAB path,
%    and contains a board definition file

boardList = { ...
    'DE10Nano.plugin_board', ...  % Path to your plugin_board.m file (as a package)
    };

% Set the workflow to DeepLearningProcessor
workflow = hdlcoder.Workflow.DeepLearningProcessor;
end