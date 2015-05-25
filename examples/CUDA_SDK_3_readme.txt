To use the examples with the nVidia GPU Computing SDK 3, make a new environmental:

NVSDKCUDA_ROOT=%NVSDKCOMPUTE_ROOT%/C

How to do this?
Control Panel->System->Advanced->Environment variables->New
Variable name: NVSDKCUDA_ROOT
Variable value: %NVSDKCOMPUTE_ROOT%/C


For debugging also add
;%NVSDKCUDA_ROOT%/bin/win32/Debug
to the path.