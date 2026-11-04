# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 yamahubuki, ACT Laboratory
"""ProgressDialog の再入 (wx.ProgressDialog.Update の中で予約済みの処理が動く) を模した試験。

本物の Update は中で yield するので、後から wx.CallAfter した close や完了メッセージが Update の途中で動く。
偽の wx で同じ状況を作り、Update の途中でダイアログが破棄されないこと、閉じてから完了メッセージが出ることを確かめる。
"""

import ast
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
queue = []
events = []


class FakeProgressDialog:
    def __init__(self, *a, **k):
        self.destroyed = False

    def CentreOnScreen(self):
        pass

    def Raise(self):
        pass

    def Update(self, pct, msg):
        events.append(("Update-start", pct))
        # 本物と同じく、Update の途中で予約済みの処理を動かす (yield)
        while queue:
            queue.pop(0)()
        if self.destroyed:
            events.append(("CRASH: Update returned into a destroyed dialog", pct))
        events.append(("Update-end", pct))
        return True, False

    def Hide(self):
        pass

    def Destroy(self):
        self.destroyed = True
        events.append(("Destroy",))


wx = types.SimpleNamespace(ProgressDialog=FakeProgressDialog, PD_CAN_ABORT=1, PD_ELAPSED_TIME=2, PD_REMAINING_TIME=4,
                           PD_AUTO_HIDE=8, CallAfter=lambda f, *a: queue.append(lambda: f(*a)),
                           CallLater=lambda ms, f: None)
gui = types.SimpleNamespace(mainFrame=types.SimpleNamespace(prePopup=lambda: None, postPopup=lambda: None))


def load_class(source_path, name):
    src = Path(source_path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    ns = {"wx": wx, "gui": gui, "threading": threading}
    exec(compile(ast.Module(body=[node], type_ignores=[]), source_path, "exec"), ns)
    return ns[name]


def scenario(PD, use_new_api):
    queue.clear()
    events.clear()
    dlg = PD("t", "m")
    if use_new_api:
        # 実際のダウンロードのスレッドと同じ: 更新を予約 → close_and_wait (閉じるまで待つ) → 完了メッセージを予約
        def worker():
            dlg.update_later(50, "a")
            dlg.update_later(99, "b")
            dlg.close_and_wait()
            wx.CallAfter(lambda: events.append(("message-box",)))

        t = threading.Thread(target=worker)
        t.start()
        # メインスレッドのイベントループ (予約された処理を順に動かす)
        while t.is_alive() or queue:
            if queue:
                queue.pop(0)()
        return list(events)
    else:
        wx.CallAfter(dlg.update, 50, "a")
        wx.CallAfter(dlg.update, 99, "b")
        wx.CallAfter(dlg.close)
    wx.CallAfter(lambda: events.append(("message-box",)))
    while queue:
        queue.pop(0)()
    return list(events)


New = load_class(ROOT / "addon/globalPlugins/DASH/__init__.py", "ProgressDialog")
print("new:", scenario(New, True))
crashed = any("CRASH" in e[0] for e in events)
order = [e[0] for e in events]
ordered = "Destroy" in order and "message-box" in order and order.index("Destroy") < order.index("message-box")
print("  crash:", crashed, "| destroy before message:", ordered)
failed = crashed or not ordered


# 修正前のクラス (再現確認用)
class Old:
    def __init__(self, title, message):
        self.cancelled = False
        self._dlg = wx.ProgressDialog(title, message)

    def update(self, pct, message):
        if self._dlg is None:
            return
        cont, _skip = self._dlg.Update(min(pct, 99), message)

    def close(self):
        if self._dlg is None:
            return
        self._dlg.Hide()
        self._dlg.Destroy()
        self._dlg = None


print("old:", scenario(Old, False))
old_crashed = any("CRASH" in e[0] for e in events)
print("  crash:", old_crashed, "(expected: the fake wx must reproduce the original crash)")
failed = failed or not old_crashed  # 偽の wx が再入を再現できていなければ、この試験は意味がない
print("FAILED" if failed else "OK")
sys.exit(1 if failed else 0)
