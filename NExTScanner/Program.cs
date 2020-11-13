using System;
using System.Collections.Generic;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    static class Program
    {
        /// <summary>
        /// The main entry point for the application.
        /// Must be STAThread to use OLE dialogs.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new SySalMainForm());
        }
    }
}