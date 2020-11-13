using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SySal.Executables.NExTScanner
{
    public interface ISySalCameraDisplay
    {
        bool EnableAutoRefresh
        {
            get;
            set;
        }

        SySal.Imaging.LinearMemoryImage ImageShown
        {
            set;
        }

        bool EnableCross
        {
            get;
            set;
        }
    }
}
