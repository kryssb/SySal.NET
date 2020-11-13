using System;
using System.Collections.Generic;
using System.Text;
using System.Xml.Serialization;

namespace SySal.ImageProcessorDisplay
{
    [Serializable]
    [XmlType("SySal.ImageProcessorDisplay.Configuration")]
    public class Configuration : SySal.Management.Configuration, ICloneable
    {
        public int PanelWidth;
        public int PanelHeight;
        public int PanelLeft;
        public int PanelTop;

        public Configuration() : base("") { }

        public Configuration(string name) : base(name) { }

        #region ICloneable Members

        public override object Clone()
        {
            SySal.ImageProcessorDisplay.Configuration C = new Configuration();
            C.PanelWidth = this.PanelWidth;
            C.PanelHeight = this.PanelHeight;
            C.PanelLeft = this.PanelLeft;
            C.PanelTop = this.PanelTop;
            return C;
        }

        #endregion

        public override string ToString()
        {
            return "Display Configuration\r\nHostedWidth = " + PanelWidth + "\r\nHostedHeight = " + PanelHeight + "\r\nPanelLeft = " + PanelLeft + "\r\nPanelTop = " + PanelTop;
        }
    }
}
