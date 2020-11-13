using System;
using System.Collections.Generic;
using System.Text;
using System.Xml.Serialization;

namespace SySal.Executables.NExTScanner
{
    [Serializable]
    [XmlType("SySal.Executables.NExTScanner.CameraDisplaySettings")]
    public class CameraDisplaySettings : SySal.Management.Configuration, ICloneable
    {
        public int PanelWidth;
        public int PanelHeight;
        public int PanelLeft;
        public int PanelTop;

        public CameraDisplaySettings() : base("") { }

        public CameraDisplaySettings(string name) : base(name) { }

        #region ICloneable Members

        public override object Clone()
        {
            SySal.Executables.NExTScanner.CameraDisplaySettings C = new CameraDisplaySettings();
            C.PanelWidth = this.PanelWidth;
            C.PanelHeight = this.PanelHeight;
            C.PanelLeft = this.PanelLeft;
            C.PanelTop = this.PanelTop;
            return C;
        }

        #endregion

        public override string ToString()
        {
            return "Camera Display Settings\r\nHostedWidth = " + PanelWidth + "\r\nHostedHeight = " + PanelHeight + "\r\nPanelLeft = " + PanelLeft + "\r\nPanelTop = " + PanelTop;
        }
    }
}
