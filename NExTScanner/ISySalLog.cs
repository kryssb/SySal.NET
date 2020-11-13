using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SySal.Executables.NExTScanner
{
    public interface ISySalLog
    {
        void Log(string error, string details);
    }
}
