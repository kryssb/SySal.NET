using System;
using System.Collections;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;
using System.Xml;
using System.Xml.Serialization;
using SySal;
using SySal.BasicTypes;
using SySal.Scanning.PostProcessing.PatternMatching;
using SySal.DAQSystem.Drivers;

namespace SySal.DAQSystem.Drivers.CSScanDriver
{
    class ZoneManagement
    {
        //private static SySal.DAQSystem.Scanning.ZoneDesc[] GetRegionScans(SySal.DAQSystem.Scanning.ZoneDesc newZones, SySal.DAQSystem.Scanning.ZoneDesc[] oldZones)
        //{
        //    if (oldZones.Length == 1) return GetRegionScans(newZones, oldZones[0]);

        //    foreach (SySal.DAQSystem.Scanning.ZoneDesc o in oldZones)
        //    {
        //        newZones = GetRegionScans(newZones, o);
        //    }
        //    return newZones;
        //}

        private static SySal.DAQSystem.Scanning.ZoneDesc[] GetRegionScans(SySal.DAQSystem.Scanning.ZoneDesc newZone, SySal.DAQSystem.Scanning.ZoneDesc oldZone)
        {
            float scale = 1000;

            float x = Convert.ToSingle(newZone.MinX / scale);
            float y = Convert.ToSingle(newZone.MinY / scale);
            float width = Convert.ToSingle((newZone.MaxX - newZone.MinX) / scale);
            float height = Convert.ToSingle((newZone.MaxY - newZone.MinY) / scale);
            System.Drawing.RectangleF ZoneToScan = new System.Drawing.RectangleF(x, y, width, height);

            x = Convert.ToSingle(oldZone.MinX / scale);
            y = Convert.ToSingle(oldZone.MinY / scale);
            width = Convert.ToSingle((oldZone.MaxX - oldZone.MinX) / scale);
            height = Convert.ToSingle((oldZone.MaxY - oldZone.MinY) / scale);
            System.Drawing.RectangleF OldZone = new System.Drawing.RectangleF(x, y, width, height);

            Console.WriteLine("ZoneToScan: " + ZoneToScan.Left + " " + ZoneToScan.Right + " " + ZoneToScan.Top + " " + ZoneToScan.Bottom);
            Console.WriteLine("OldZone:    " + OldZone.Left + " " + OldZone.Right + " " + OldZone.Top + " " + OldZone.Bottom);

            System.Drawing.Region NewRegion = new System.Drawing.Region(ZoneToScan);

            NewRegion.Exclude(OldZone);

            System.Drawing.RectangleF[] rectangles = NewRegion.GetRegionScans(new System.Drawing.Drawing2D.Matrix());

            if (rectangles.Length == 0) return null;

            SySal.DAQSystem.Scanning.ZoneDesc[] zones = new SySal.DAQSystem.Scanning.ZoneDesc[rectangles.Length];

            for (int i = 0; i < zones.Length; i++)
            {
                zones[i] = new SySal.DAQSystem.Scanning.ZoneDesc();
                zones[i].MinX = rectangles[i].Left * scale;
                zones[i].MaxX = rectangles[i].Right * scale;
                zones[i].MinY = rectangles[i].Top * scale;
                zones[i].MaxY = rectangles[i].Bottom * scale;
            }
            return zones;
        }
    }
}