// Emo (React Native) — type a task and watch on-device emoji suggestions update
// live. The model runs fully offline from bundled assets; see emo.js.
import { useEffect, useState } from "react";
import { ActivityIndicator, SafeAreaView, StyleSheet, Text, TextInput, View } from "react-native";

import { suggestions } from "./emo";

export default function App() {
  const [text, setText] = useState("");
  const [emojis, setEmojis] = useState([]);
  const [ready, setReady] = useState(false);

  // Warm up the model once (loads bundled assets — no network).
  useEffect(() => {
    suggestions("hello", 1).finally(() => setReady(true));
  }, []);

  // Debounce predictions while typing.
  useEffect(() => {
    if (!text.trim()) {
      setEmojis([]);
      return;
    }
    let cancelled = false;
    const t = setTimeout(async () => {
      const results = await suggestions(text, 3);
      if (!cancelled) setEmojis(results.map((r) => r.emoji));
    }, 150);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
  }, [text]);

  return (
    <SafeAreaView style={styles.root}>
      <Text style={styles.title}>Emo · on-device emoji</Text>
      <TextInput
        style={styles.input}
        placeholder="Type a task…"
        value={text}
        onChangeText={setText}
        autoFocus
        autoCorrect={false}
      />
      <View style={styles.row}>
        {ready ? (
          emojis.map((e, i) => (
            <Text key={i} style={styles.emoji}>
              {e}
            </Text>
          ))
        ) : (
          <ActivityIndicator />
        )}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, padding: 24, gap: 16, backgroundColor: "#fff", justifyContent: "center" },
  title: { fontSize: 20, fontWeight: "600", textAlign: "center" },
  input: { borderWidth: 1, borderColor: "#ddd", borderRadius: 12, padding: 14, fontSize: 18 },
  row: { flexDirection: "row", gap: 16, justifyContent: "center", minHeight: 48, alignItems: "center" },
  emoji: { fontSize: 40 },
});
