using System;

namespace SySal.Executables.NExTScanner
{
    public interface IMapProvider
    {
        SySal.DAQSystem.Scanning.IntercalibrationInfo PlateMap
        {
            get;
        }

        SySal.DAQSystem.Scanning.IntercalibrationInfo InversePlateMap
        {
            get;
            set;
        }
    }
}
