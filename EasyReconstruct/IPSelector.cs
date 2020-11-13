using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.EasyReconstruct
{
    /// <summary>
    /// Selects tracks/vertices for IP computation.
    /// </summary>
    public interface IPSelector
    {
        /// <summary>
        /// Selects a track as first or second for IP computation.
        /// </summary>
        /// <param name="tk">the track to be selected.</param>
        /// <param name="isfirst"><c>true</c> if the track is the first for IP computation, <c>false</c> if it is the second.</param>
        void SelectTrack(SySal.TotalScan.Track tk, bool isfirst);
        /// <summary>
        /// Selects a vertex for IP computation.
        /// </summary>
        /// <param name="vtx">the vertex to be selected.</param>
        void SelectVertex(SySal.TotalScan.Vertex vtx);
        /// <summary>
        /// Sets the label of an object.
        /// </summary>
        /// <param name="owner">the object whose label should be set.</param>
        /// <param name="label">the label text to be set.</param>
        void SetLabel(object owner, string label);
        /// <summary>
        /// Shows/hides label for an object.
        /// </summary>
        /// <param name="owner">the object whose label should be shown/hidden.</param>
        /// <param name="enable"><c>true</c> to show the label, <c>false</c> to hide it.</param>
        void EnableLabel(object owner, bool enable);
        /// <summary>
        /// Sets the highlighting status for an object.
        /// </summary>
        /// <param name="owner">the object whose highlighting status must be set.</param>
        /// <param name="hlstatus"><c>true</c> to highlight, <c>false</c> to show in normal fashion.</param>
        void Highlight(object owner, bool hlstatus);
        /// <summary>
        /// Removes an object from the display.
        /// </summary>
        /// <param name="owner">the object to be deleted.</param>
        /// <returns><c>true</c> if the object has been found and deleted, <c>false</c> if it was not contained in the plot.</returns>
        bool DeleteWithOwner(object owner);
        /// <summary>
        /// Adds an object (track) to the display.
        /// </summary>
        /// <param name="owner">the object to be added.</param>        
        void Plot(object owner);
    }
}
