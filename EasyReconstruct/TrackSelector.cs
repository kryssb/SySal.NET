using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.EasyReconstruct
{
    /// <summary>
    /// A generic delegate to be called on each track to check whether it must be shown or not.
    /// </summary>
    /// <param name="t">the track being examined.</param>
    /// <returns><c>true</c> if the track must be shown, <c>false</c> otherwise</returns>
    public delegate bool dShow(SySal.TotalScan.Track t);

    /// <summary>
    /// A generic delegate to be called on each segment to check whether it must be shown or not.
    /// </summary>
    /// <param name="s">the segment being examined.</param>
    /// <returns><c>true</c> if the track must be shown, <c>false</c> otherwise</returns>
    public delegate bool dShowSeg(SySal.TotalScan.Segment s);

    /// <summary>
    /// A generic delegate for GUI events.
    /// </summary>
    /// <param name="sender">the event sender.</param>
    /// <param name="e">event arguments.</param>
    public delegate void dGenericEvent(object sender, EventArgs e);    

    /// <summary>
    /// Selects tracks as graphical elements.
    /// </summary>
    public interface TrackSelector
    {
        /// <summary>
        /// Adds track to the graph, using the specified filter. A null filter selects all tracks.
        /// </summary>
        void ShowTracks(dShow filter, bool show, bool makedataset);

        /// <summary>
        /// Adds track to the graph, using the specified filter. A null filter selects all tracks.
        /// </summary>
        void ShowSegments(dShowSeg filter, bool show, bool makedataset);

        /// <summary>
        /// Adds the segments in the list to the graph. <c>null</c> values are skipped.
        /// </summary>
        /// <param name="segs">the list of segments to be added to the graph.</param>
        /// <param name="owner">the owner to which ancillary graphical elements must be attached.</param>
        void ShowSegments(SySal.TotalScan.Segment[] segs, object owner);

        /// <summary>
        /// Finds segments in a scanback/scanforth path with respect to a starting segment.
        /// </summary>
        /// <param name="start">segment to start from.</param>
        /// <param name="mingrains">minimum number of grains.</param>
        /// <param name="postol">position tolerance for scanback/scanforth.</param>
        /// <param name="slopetol">transverse slope tolerance (or longitudinal slope tolerance at 0 slope).</param>
        /// <param name="slopetolincrease">longitudinal slope tolerance increase coefficient. The actual longitudinal tolerance is <c>slope * slopetolincrease + slopetol</c>.</param>
        /// <param name="maxmiss">maximum number of consecutive layers where no related track is found.</param>
        /// <param name="downstream"><c>true</c> for Scanforth, <c>false</c> for Scanback.</param>
        /// <returns>a list of segments found related with the starting one.</returns>
        SySal.TotalScan.Segment[] Follow(SySal.TotalScan.Segment start, bool downstream, int mingrains, double postol, double slopetol, double slopetolincrease, int maxmiss);

        /// <summary>
        /// Subscribe to OnAddFit.
        /// </summary>
        /// <param name="ge">the delegate to be added.</param>
        void SubscribeOnAddFit(dGenericEvent ge);

        /// <summary>
        /// Remove subscription to OnAddFit.
        /// </summary>
        /// <param name="ge">the delegate to be removed.</param>
        void UnsubscribeOnAddFit(dGenericEvent ge);

        /// <summary>
        /// Trigger the AddFit event.
        /// </summary>
        /// <param name="sender">the object that raises the event.</param>
        /// <param name="e">the parameters of the event.</param>
        void RaiseAddFit(object sender, EventArgs e);

        /// <summary>
        /// Toggles addition of selected segments.
        /// </summary>
        /// <param name="so">when <c>null</c>, stops adding new segments; otherwise, points to the method that should be called to add segments.</param>
        void ToggleAddReplaceSegments(GDI3D.Control.SelectObject so);

        /// <summary>
        /// Notifies that a track is added.
        /// </summary>
        /// <param name="tk">the track to be added.</param>
        void Add(SySal.TotalScan.Track tk);

        /// <summary>
        /// Notifies that a track is removed.
        /// </summary>
        /// <param name="tk">the track to be removed.</param>
        void Remove(SySal.TotalScan.Track tk);

        /// <summary>
        /// Notifies that a vertex is removed.
        /// </summary>
        /// <param name="tk">the track to be removed.</param>
        void Remove(SySal.TotalScan.Vertex vx);
    }
}
